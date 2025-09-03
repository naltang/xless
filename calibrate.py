
#!/usr/bin/env python3
"""
Image calibration by rank‑1 factorisation

Author:  <your name>
Date:    2025‑09‑03

The script reads all *.raw files in a folder, assumes that every file is
2560×2048 pixels (16 bit unsigned little endian).  
It finds the best pixel‑wise distortion vector d and image‑specific gains g
that minimise the mean‑squared error

    MSE = 1/(M·N) Σ_k,j ( X_kj – g_k · d_j )²

where X is the raw data matrix, M = #images and N = 2560×2048.

The result is written to two CSV files:

* `d_matrix.csv`      – one row per image height, comma separated values.
* `g_values.csv`      – each line: <raw‑file-name>,<gain>

Dependencies
-------------
    numpy   >= 1.20
    torch   >= 2.0

Usage
-----
    python calibrate.py /path/to/raw_folder
"""

import os
from pathlib import Path
import sys

import numpy as np
import torch


# ------------------------------------------------------------
#  Parameters – change only if the sensor size is different.
HEIGHT = 2560          # image height
WIDTH  = 2048          # image width
PIXELS = HEIGHT * WIDTH
DTYPE  = "<u2"         # raw files are little‑endian unsigned short
# ------------------------------------------------------------


def read_raw_file(path: Path) -> np.ndarray:
    """
    Read a single .raw file and return it as a (H,W) float32 array.
    """
    data = np.fromfile(path, dtype=DTYPE)
    if data.size != PIXELS:
        raise ValueError(f"File {path} has size {data.size*2/1024**2:.1f} MiB "
                         f"but should be {PIXELS*2/1024**2:.1f} MiB")
    return data.astype(np.float32).reshape(HEIGHT, WIDTH)


def main(folder: str):
    folder = Path(folder)
    raw_files = sorted(folder.glob("*.raw"))
    if not raw_files:
        raise RuntimeError(f"No *.raw files found in {folder}")

    M = len(raw_files)          # number of images
    print(f"Found {M} RAW images ({HEIGHT}×{WIDTH})")

    # ------------------------------------------------------------
    #  Load all images into a single matrix X (M × N)
    # ------------------------------------------------------------
    X = np.empty((M, PIXELS), dtype=np.float32)

    for i, fp in enumerate(raw_files):
        img = read_raw_file(fp).flatten()
        X[i] = img

    print("All images loaded – performing rank‑1 factorisation...")

    # ------------------------------------------------------------
    #  Rank‑1 SVD with PyTorch (GPU is optional)
    # ------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X_torch = torch.from_numpy(X).to(device)

    U, S, Vh = torch.linalg.svd(X_torch, full_matrices=False)
    sigma0   = S[0].item()
    u0       = U[:, 0]          # shape (M,)
    v0       = Vh[0]            # shape (N,)

    # pixel distortion (d) and image gain (g)
    d_flat = np.sqrt(sigma0) * v0.cpu().numpy()   # length N
    g_vec  = np.sqrt(sigma0) * u0.cpu().numpy()   # length M

    # ------------------------------------------------------------
    #  Fix the scale: make mean(d)=1 (arbitrary but convenient)
    # ------------------------------------------------------------
    mean_d = d_flat.mean()
    d_scaled = d_flat / mean_d
    g_scaled = g_vec * mean_d

    d_matrix = d_scaled.reshape(HEIGHT, WIDTH)

    # ------------------------------------------------------------
    #  Write the two CSV files
    # ------------------------------------------------------------
    np.savetxt("d_matrix.csv", d_matrix,
               delimiter=",", fmt="%.6f")
    print("Saved distortion matrix to 'd_matrix.csv'")

    with open("g_values.csv", "w") as f:
        for fname, g in zip([p.name for p in raw_files], g_scaled):
            f.write(f"{fname},{g:.6f}\n")
    print("Saved image gains to 'g_values.csv'")

    # Optional: compute MSE
    pred = np.outer(g_scaled, d_scaled)
    mse = ((X - pred)**2).mean()
    print(f"Finished. Rank‑1 MSE = {mse:.3e}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Calibrate raw images using rank-1 factorisation.")
    parser.add_argument("folder", help="Folder containing .raw files")
    args = parser.parse_args()
    try:
        main(args.folder)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
