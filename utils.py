


def save_backbone_pdb(
        coords,
        masks,
        save_path_prefix,
        atom_names=("N", "CA", "C"),
        chain_id="A",
):
    """
    Write backbone (N, CA, C) atom coordinates to PDB files—one file per item in the batch—
    with every field strictly aligned to the official PDB column specification.

    Parameters
    ----------
    coords : torch.Tensor
        Shape (B, L, 3, 3) or (L, 3, 3).  Last two axes are atoms × (x,y,z).
    masks : torch.Tensor
        Shape (B, L) or (L,).  1 → keep residue, 0 → skip residue.
    save_path_prefix : str
        Path prefix.  “_sample_<idx>.pdb” is appended (or inserted before “.pdb”).
    atom_names : tuple[str], default ("N", "CA", "C")
        Three backbone atom names, in the same order as coords[..., atom, :].
    chain_id : str, default "A"
        Single-letter chain identifier.
    """
    import torch  # only needed for `coords.dim()`

    # Ensure batch dimension exists
    if coords.dim() == 3:
        coords = coords.unsqueeze(0)
        masks = masks.unsqueeze(0)

    B, L = coords.shape[:2]

    for b in range(B):
        # Build output file name
        if save_path_prefix.lower().endswith(".pdb"):
            root = save_path_prefix[:-4]
            out_path = f"{root}.pdb"
        else:
            out_path = f"{save_path_prefix}.pdb"

        with open(out_path, "w") as fh:
            serial = 1
            for r in range(L):
                if masks[b, r].item() != 1:
                    continue

                for a_idx, atom_name in enumerate(atom_names):
                    x, y, z = coords[b, r, a_idx].tolist()
                    element = atom_name[0].upper()

                    # ┌────────────────────────────────────────── columns ──────────────────────────────────────────┐
                    #  1–6  "ATOM  "
                    #  7–11 serial  (right-justified 5-digit)
                    #    12 blank
                    # 13–16 atom name (right-justified 4)
                    #    17 altLoc   (blank)
                    # 18–20 resName  ("UNK")
                    #    21 blank
                    #    22 chainID
                    # 23–26 resSeq   (right-justified 4)
                    #    27 iCode    (blank)
                    # 28–30 blanks   (3)   ← keeps x in col 31
                    # 31–38 x (8.3f)
                    # 39–46 y (8.3f)
                    # 47–54 z (8.3f)
                    # 55–60 occupancy (6.2f)
                    # 61–66 tempFactor (6.2f)
                    # 67–76 blanks (10)
                    # 77–78 element (right-justified 2)
                    # └────────────────────────────────────────────────────────────────────────────────────────────┘
                    fh.write(
                        f"ATOM  "
                        f"{serial:5d} "
                        f"{atom_name:>4s}"
                        f" "
                        f"UNK"
                        f" "
                        f"{chain_id}"
                        f"{r + 1:4d}"
                        f" "
                        f"   "
                        f"{x:8.3f}"
                        f"{y:8.3f}"
                        f"{z:8.3f}"
                        f"{1.00:6.2f}"
                        f"{0.00:6.2f}"
                        f"          "
                        f"{element:>2s}"
                        "\n"
                    )
                    serial += 1

            fh.write("TER\nEND\n")