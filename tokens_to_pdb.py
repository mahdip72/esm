import torch
from torch.utils.data import Dataset, DataLoader
from esm.pretrained import ESM3_structure_decoder_v0
from esm.tokenization.structure_tokenizer import StructureTokenizer
from pathlib import Path
from tqdm import tqdm
import warnings
import csv
import os
from torch.cuda.amp import autocast
import functools # Import functools
from utils import save_backbone_pdb # Added import

# Suppress specific UserWarnings from biotite
warnings.filterwarnings("ignore", message=r".*elements were guessed from atom name.*", category=UserWarning)

# Configuration
INPUT_CSV_FILE = "data/cameo.csv"  # Input CSV file path
OUTPUT_PDB_DIR = "/mnt/hdd8/mehdi/projects/FoldToken_open/foldtoken/results/reconstructions/cameo_esm3"  # Directory to save reconstructed PDBs
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2  # Adjust based on your GPU memory
NUM_WORKERS = 8 # Set to 0 for debugging


class TokenCSVDataset(Dataset):
    def __init__(self, csv_file_path):
        self.data = []
        try:
            with open(csv_file_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        tokens = [int(t) for t in row['discrete_tokens'].split(',')]
                        protein_length = int(row['length'])  # This now includes BOS/EOS tokens

                        self.data.append({
                            'pdb_name': row['pdb_name'],
                            'tokens': tokens,
                            'length': protein_length,
                        })
                    except ValueError as e:
                        print(f"Skipping row for {row.get('pdb_name', 'Unknown PDB')} due to data parsing error: {e}")
                    except KeyError as e:
                        print(f"Skipping row due to missing key: {e} in CSV for {row.get('pdb_name', 'Unknown PDB')}")
        except FileNotFoundError:
            print(f"Error: Input CSV file not found at {csv_file_path}")
            # self.data remains empty, len(self) will be 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def collate_fn_for_decoding(batch_items, tokenizer): # Removed device argument
    if not batch_items: # Handle cases where a batch might be empty if all items in TokenCSVDataset failed to load (though unlikely with current setup)
        return None

    pdb_names = [item['pdb_name'] for item in batch_items]
    original_lengths = [item['length'] for item in batch_items] # Length including BOS/EOS tokens

    max_len_struct = max(original_lengths) if original_lengths else 0

    struct_tokens_batch_list = []
    decoder_mask_batch_list = []

    for item in batch_items:
        structure_tokens = item['tokens']  # These already include BOS/EOS
        current_len_struct = item['length']

        # Pad structure tokens (which already include BOS/EOS)
        padded_item_tokens = torch.full((max_len_struct,), tokenizer.pad_token_id, dtype=torch.long)
        if current_len_struct > 0: # Ensure there are tokens to place
            padded_item_tokens[:current_len_struct] = torch.tensor(structure_tokens, dtype=torch.long)
        # No need to add BOS/EOS manually - they're already in structure_tokens

        struct_tokens_batch_list.append(padded_item_tokens)

        item_decoder_mask = torch.zeros(max_len_struct, dtype=torch.bool)
        if current_len_struct > 0:
            item_decoder_mask[:current_len_struct] = True
        decoder_mask_batch_list.append(item_decoder_mask)

    if not struct_tokens_batch_list: # If all items had issues or were empty
        return None

    struct_tokens_batch = torch.stack(struct_tokens_batch_list) # Keep on CPU
    decoder_mask_batch = torch.stack(decoder_mask_batch_list)   # Keep on CPU

    return {
        'struct_tokens_batch': struct_tokens_batch,
        'decoder_mask_batch': decoder_mask_batch,
        'pdb_names_list': pdb_names,
        'original_lengths_list': original_lengths,
    }


def main():
    device_obj = torch.device(DEVICE)
    print(f"Using device: {device_obj}")

    print("Loading structure tokenizer...")
    st_tokenizer = StructureTokenizer()
    print("Loading structure decoder...")
    decoder = ESM3_structure_decoder_v0(device=device_obj)
    decoder = decoder.half()
    decoder.eval()

    output_pdb_path = Path(OUTPUT_PDB_DIR)
    output_pdb_path.mkdir(parents=True, exist_ok=True)

    dataset = TokenCSVDataset(INPUT_CSV_FILE)
    if not dataset or len(dataset) == 0: # Check if dataset is empty after initialization
        print(f"No data loaded from {INPUT_CSV_FILE} or dataset is empty. Exiting.")
        return

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=functools.partial(collate_fn_for_decoding, tokenizer=st_tokenizer), # Use functools.partial
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True if device_obj.type == 'cuda' else False # Added pin_memory for potential speedup
    )

    print(f"Processing {len(dataset)} entries from CSV for PDB reconstruction...")
    for batch_data in tqdm(dataloader, desc="Decoding to PDBs"):
        if batch_data is None:
            print("Skipping an empty batch from collate_fn.")
            continue

        struct_tokens = batch_data['struct_tokens_batch'].to(device_obj) # Moved to device here
        decoder_mask = batch_data['decoder_mask_batch'].to(device_obj)   # Moved to device here
        pdb_names = batch_data['pdb_names_list']
        original_lengths = batch_data['original_lengths_list']

        on_cuda = device_obj.type == 'cuda'
        decoded_batch = None

        with torch.no_grad():
            if on_cuda:
                with autocast(dtype=torch.float16):
                    decoded_batch = decoder.decode(struct_tokens, attention_mask=decoder_mask)
            else:
                # On CPU, ensure model and inputs are compatible if using .half()
                # For simplicity, assuming decoder.decode handles types or CPU .half() is fine.
                # struct_tokens is LongTensor, which is fine.
                decoded_batch = decoder.decode(struct_tokens, attention_mask=decoder_mask)

        if decoded_batch is None or 'bb_pred' not in decoded_batch:
            print("Decoding failed or 'bb_pred' not in output. Skipping batch.")
            continue

        bb_pred_batch = decoded_batch.get('bb_pred')

        for i in range(len(pdb_names)):
            pdb_name = pdb_names[i]
            current_struct_length = original_lengths[i]  # This includes BOS/EOS tokens
            current_vq_length = current_struct_length - 2  # Subtract BOS/EOS to get actual residue count

            if current_struct_length <= 2:  # Only BOS/EOS tokens
                print(f"Skipping PDB generation for {pdb_name} due to no actual residues (only BOS/EOS).")
                continue

            # Extract coordinates between BOS and EOS tokens (positions 1 to current_vq_length+1)
            new_coords_bb = bb_pred_batch[i, 1:current_vq_length+1, :, :]

            if new_coords_bb.size(0) != current_vq_length:
                print(f"Warning: Length mismatch for {pdb_name}. Expected {current_vq_length}, got {new_coords_bb.size(0)}. Skipping save.")
                continue

            # Use save_backbone_pdb instead
            try:
                # Ensure coordinates and mask are on CPU for PDB writing
                coords_for_pdb = new_coords_bb.cpu() 
                mask_for_pdb = torch.ones(current_vq_length, device=coords_for_pdb.device, dtype=torch.bool) # Function expects boolean or 0/1

                # Construct the save path prefix.
                # save_backbone_pdb will append _sample_0.pdb (or similar)
                file_prefix = str(output_pdb_path / pdb_name)

                save_backbone_pdb(
                    coords=coords_for_pdb,
                    masks=mask_for_pdb,
                    save_path_prefix=file_prefix
                    # atom_names and chain_id will use defaults ("N", "CA", "C") and "A"
                )
                # Note: The actual saved filename will be like "{file_prefix}_sample_0.pdb"
                # print(f"Saved backbone PDB for {pdb_name} (as {file_prefix}_sample_0.pdb)") # Optional: for more specific logging
            except Exception as e:
                print(f"Error saving backbone PDB for {pdb_name} using prefix {file_prefix}: {e}")

    print(f"Processing complete. Reconstructed PDBs saved to {output_pdb_path}")

if __name__ == "__main__":
    main()
