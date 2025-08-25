import torch
from torch.utils.data import Dataset, DataLoader
from esm.sdk.api import ESMProtein
from esm.pretrained import ESM3_structure_encoder_v0
from esm.tokenization.structure_tokenizer import StructureTokenizer # Now needed for BOS/EOS tokens
from pathlib import Path
from tqdm import tqdm
import warnings
import csv
import os
from torch.cuda.amp import autocast # Add this import

# Suppress specific UserWarnings from biotite
warnings.filterwarnings("ignore", message=r".*elements were guessed from atom name.*", category=UserWarning)

# Configuration
INPUT_DIR = "/mnt/hdd8/mehdi/projects/FoldToken_open/foldtoken/results/originals/cameo"  # Update if necessary, e.g., data/T493
OUTPUT_CSV_FILE = "data/cameo.csv"  # Output CSV file path
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu" # Default to cuda:0
BATCH_SIZE = 8  # Adjust based on your GPU memory
NUM_WORKERS = 4 # Adjust based on your system configuration


class PDBDataset(Dataset):
    def __init__(self, pdb_files):
        self.pdb_files = [f for f in pdb_files if f.is_file() and f.suffix == '.pdb'] # Ensure only .pdb files are processed

    def __len__(self):
        return len(self.pdb_files)

    def __getitem__(self, idx):
        pdb_path = self.pdb_files[idx]
        try:
            protein = ESMProtein.from_pdb(str(pdb_path))
        except Exception as e:
            print(f"Skipping {pdb_path.name} due to loading error: {e}")
            return None

        if protein.coordinates is None or len(protein) == 0:
            print(f"Skipping {pdb_path.name} due to no coordinates or zero length.")
            return None

        coords = protein.coordinates
        original_length = len(protein)
        sequence = protein.sequence # Extract sequence

        encoder_mask = torch.ones(original_length, dtype=torch.bool) # Default if no specific mask
        if hasattr(protein, 'padding_mask') and protein.padding_mask is not None:
            encoder_mask = ~protein.padding_mask.bool()

        residx_tensor = None
        if hasattr(protein, 'residue_indices') and protein.residue_indices is not None:
            residx_tensor = torch.tensor(protein.residue_indices, dtype=torch.long) if not isinstance(protein.residue_indices, torch.Tensor) else protein.residue_indices.long()
        elif hasattr(protein, 'residx') and protein.residx is not None:
            residx_tensor = torch.tensor(protein.residx, dtype=torch.long) if not isinstance(protein.residx, torch.Tensor) else protein.residx.long()


        return {
            'coordinates': coords,
            'encoder_mask': encoder_mask,
            'residx': residx_tensor,
            'original_length': original_length,
            'pdb_name': pdb_path.name,
            'sequence': sequence, # Add sequence to the output
        }

def collate_fn_for_encoding(batch_items_with_potential_nones):
    batch_items = [item for item in batch_items_with_potential_nones if item is not None]
    if not batch_items:
        return None

    pdb_names = [item['pdb_name'] for item in batch_items]
    original_lengths_list = [item['original_length'] for item in batch_items]
    original_lengths = torch.tensor(original_lengths_list, dtype=torch.long)
    sequences = [item['sequence'] for item in batch_items] # Collect sequences

    coords_list = [item['coordinates'] for item in batch_items]
    max_len_coord = max(c.size(0) for c in coords_list)

    padded_coords_list = []
    for c in coords_list:
        padding_size = max_len_coord - c.size(0)
        padded_c = torch.nn.functional.pad(c, (0,0, 0,0, 0,padding_size), value=0.0)
        padded_coords_list.append(padded_c)
    padded_coords = torch.stack(padded_coords_list)

    encoder_masks_list = [item['encoder_mask'] for item in batch_items]
    padded_encoder_masks_list = []
    for m in encoder_masks_list:
        padding_size = max_len_coord - m.size(0)
        padded_m = torch.nn.functional.pad(m, (0, padding_size), value=False)
        padded_encoder_masks_list.append(padded_m)
    padded_encoder_masks = torch.stack(padded_encoder_masks_list)

    padded_residx = None
    if all(item['residx'] is not None for item in batch_items):
        residx_list = [item['residx'] for item in batch_items]
        padded_residx_list = []
        for r in residx_list:
            padding_size = max_len_coord - r.size(0)
            padded_r = torch.nn.functional.pad(r, (0, padding_size), value=0)
            padded_residx_list.append(padded_r)
        padded_residx = torch.stack(padded_residx_list)
    else:
        # Create a default residx if any item is None, or handle as per encoder requirements
        # For now, assuming encoder can handle None or a default tensor if some are missing.
        # Or, ensure all PDBs provide valid residx.
        # If encoder strictly needs it, PDBs without it should be skipped or have a default generated.
        # For simplicity, if any residx is None, the whole batch residx is None.
        # This might need adjustment based on how ESM3_structure_encoder_v0 handles None residue_index.
        # Typically, residue_index is recommended for best performance.
        # Let's try to create a default if it's None and some are not.
        # However, the encoder expects either a tensor for the batch or None.
        # So if any is None, we might have to pass None for the whole batch or skip.
        # The current structure of ESM3_structure_encoder_v0 suggests it can take None.
        pass


    return {
        'coordinates_batch': padded_coords,
        'encoder_mask_batch': padded_encoder_masks,
        'residx_batch': padded_residx,
        'original_lengths': original_lengths,
        'pdb_names_list': pdb_names,
        'sequences_list': sequences, # Add sequences to collated output
    }


def main():
    device = torch.device(DEVICE)
    print(f"Using device: {device}")

    print("Loading structure encoder...")
    encoder = ESM3_structure_encoder_v0(device=device)
    encoder = encoder.half() # Convert model to float16
    encoder.eval()
    st_tokenizer = StructureTokenizer() # Needed for BOS/EOS tokens

    input_path = Path(INPUT_DIR)
    output_csv = Path(OUTPUT_CSV_FILE)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    pdb_files = list(input_path.glob("*.pdb"))
    if not pdb_files:
        print(f"No PDB files found in {INPUT_DIR}")
        return

    dataset = PDBDataset(pdb_files)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        collate_fn=collate_fn_for_encoding,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    results_list = []
    print(f"Processing {len(dataset)} PDB files...")
    for batch_data in tqdm(dataloader, desc="Encoding PDBs"):
        if batch_data is None:
            print("Skipping an empty batch.")
            continue

        coords_batch = batch_data['coordinates_batch'].to(device).to(torch.float16) # Cast input to float16
        encoder_mask_batch = batch_data['encoder_mask_batch'].to(device).bool()
        residx_batch = batch_data['residx_batch'].to(device) if batch_data['residx_batch'] is not None else None
        original_lengths = batch_data['original_lengths'] # on CPU
        pdb_names_list = batch_data['pdb_names_list']
        sequences_list = batch_data['sequences_list'] # Get sequences

        on_cuda = device.type == 'cuda'

        with torch.no_grad():
            if on_cuda:
                with autocast(dtype=torch.float16):
                    _, vq_indices_batch = encoder.encode(
                        coords_batch,
                        attention_mask=encoder_mask_batch,
                        residue_index=residx_batch
                    )
            else:
                # If on CPU, autocast is a no-op.
                # Running .half() models on CPU can sometimes lead to issues
                # if not all ops support float16. The error seems GPU-specific.
                _, vq_indices_batch = encoder.encode(
                    coords_batch,
                    attention_mask=encoder_mask_batch,
                    residue_index=residx_batch
                )
            # vq_indices_batch is (B, S_vq), where S_vq is max original_length in batch

        for i in range(coords_batch.size(0)):
            pdb_name_with_ext = pdb_names_list[i]
            pdb_name = os.path.splitext(pdb_name_with_ext)[0]
            current_original_length = original_lengths[i].item()
            current_sequence = sequences_list[i] # Get current sequence

            # Get the VQ indices for the current protein, up to its original length
            item_vq_indices = vq_indices_batch[i, :current_original_length].cpu().tolist()

            # Get the coordinate validity mask for this protein
            current_affine_mask = encoder_mask_batch[i, :current_original_length].cpu()

            # Replace VQ indices with MASK tokens where coordinates are invalid
            # This is CRITICAL: when affine_mask is False, the encoder zeros out embeddings
            # before VQ quantization, making those VQ indices meaningless
            processed_vq_indices = []
            for j, vq_idx in enumerate(item_vq_indices):
                if current_affine_mask[j]:  # Valid coordinates
                    processed_vq_indices.append(vq_idx)
                else:  # Invalid coordinates - use MASK token
                    processed_vq_indices.append(st_tokenizer.mask_token_id)

            # Add BOS and EOS tokens to match official ESM3 format
            # BOS (4098) at beginning, EOS (4097) at end, MASK (4096) for invalid coords
            # VQ indices are in range [0, 4095], special tokens are [4096, 4100]
            structure_tokens = [st_tokenizer.bos_token_id] + processed_vq_indices + [st_tokenizer.eos_token_id]
            tokens_str = ",".join(map(str, structure_tokens))
            
            # Length now includes BOS/EOS tokens (original_length + 2)
            structure_length = current_original_length + 2

            results_list.append({
                'pdb_name': pdb_name,
                'discrete_tokens': tokens_str,
                'length': structure_length, # Updated to include BOS/EOS
                'sequence': current_sequence, # Add sequence to results
            })

    # Write to CSV
    if results_list:
        fieldnames = ['pdb_name', 'discrete_tokens', 'length', 'sequence'] # Add sequence to fieldnames
        with open(output_csv, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_list)
        print(f"Processing complete. Tokenized data saved to {output_csv}")
    else:
        print("No results to save.")

if __name__ == "__main__":
    main()
