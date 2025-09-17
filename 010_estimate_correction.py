#!/usr/bin/env python3
"""
Least-squares estimation of per-pixel correction factors Cpixel and per-file gains Gimage
from 16-bit raw images.

Usage:
    python estimate_correction.py /path/to/raw_folder

The program will:
    - read all *.raw files in the supplied folder (they must have identical length)
    - compute Cpixel and Gimage that minimise Σ_i,k [RAW_i[k] * Cpixel[k] * Gimage[i] – 50000]^2
      with mean(Cpixel)=1
    - write Cpixel to correction.csv
    - print the Gimage values for all files
"""

import os
import argparse
from typing import List, Tuple

import numpy as np


def load_raw_files(folder: str, ext: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load all *.raw files in *folder* into a list of 1-D float64 arrays.
    All files must have the same length; otherwise ValueError is raised.

    Returns
    -------
    raw_arrays : list of np.ndarray (dtype=float64)
        Raw data for each file, shape (N,)
    filenames : list of str
        Corresponding filenames (without path)
    """
    raw_files = sorted(
        f for f in os.listdir(folder) if f.lower().endswith(ext)
    )
    if not raw_files:
        raise ValueError(f"No *.raw files found in folder '{folder}'")

    raw_arrays: List[np.ndarray] = []
    filenames: List[str] = []

    expected_len = None
    for fn in raw_files:
        path = os.path.join(folder, fn)
        data = np.fromfile(path, dtype=np.uint16).astype(np.float64)
        if expected_len is None:
            expected_len = len(data)
        elif len(data) != expected_len:
            raise ValueError(
                f"File '{fn}' has length {len(data)}; "
                f"expected {expected_len}"
            )
        raw_arrays.append(data)
        filenames.append(fn)

    return raw_arrays, filenames


def fit_correction_factors(
    raw_arrays: List[np.ndarray],
    max_iter: int = 100,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform alternating least-squares to estimate Cpixel and Gimage.

    Parameters
    ----------
    raw_arrays : list of np.ndarray
        Raw data for each file (shape (N,))
    max_iter : int, optional
        Maximum number of ALS iterations.
    tol : float, optional
        Convergence tolerance on the maximum change in Cpixel or Gimage.

    Returns
    -------
    Cpixel : np.ndarray shape (N,)
        Per-pixel correction factors (normalised so mean(Cpixel)=1)
    Gimage : np.ndarray shape (M,)
        Per-file gain scalars (>0)
    """
    # Convert to a 2-D array for easier vectorisation: shape (M, N)
    Y = np.stack(raw_arrays)          # dtype float64
    M, N = Y.shape

    eps = 1e-12                      # avoid division by zero

    Cpixel = np.ones(N, dtype=np.float64)

    Gimage = np.ones(M, dtype=np.float64)   # initial guess (will be updated)
    for it in range(max_iter):
        print(f"iteration {it}")
        old_Cpixel = Cpixel.copy()
        old_Gimage = Gimage.copy()

        # ---------- Update Gimage ----------
        # For each file i: minimise Σ_k [Cpixel[k]*Gimage[i]*Y[i,k] – 50000]^2
        denom_Gimage = np.sum((Cpixel * Y)**2, axis=1) + eps          # shape (M,)
        numer_Gimage = np.sum(Cpixel * Y, axis=1) * 50000.0           # shape (M,)
        Gimage = numer_Gimage / denom_Gimage

        # ---------- Update Cpixel ----------
        # For each pixel k: minimise Σ_i [Cpixel[k]*Gimage[i]*Y[i,k] – 50000]^2
        # Vectorised over all pixels:
        Gimage_col = Gimage[:, None]                                 # shape (M,1)
        denom_Cpixel = np.sum((Gimage_col * Y)**2, axis=0) + eps     # shape (N,)
        numer_Cpixel = np.sum(Gimage_col * Y, axis=0) * 50000.0      # shape (N,)
        Cpixel = numer_Cpixel / denom_Cpixel

        # Normalise Cpixel so that its mean is 1
        Cpixel /= Cpixel.mean()

        # ---------- Convergence check ----------
        delta_Cpixel = np.max(np.abs(Cpixel - old_Cpixel))
        delta_Gimage = np.max(np.abs(Gimage - old_Gimage))
        print(f"Cpixel= ... {Cpixel[N//2 + 1000 : N//2 + 1005]} ...")
        print(f"Gimage= ... {Gimage[M//2 : M//2 + 5]} ...")
        print(f"tol={tol}, delta_Cpixel={delta_Cpixel}, delta_Gimage={delta_Gimage}")
        if max(delta_Cpixel, delta_Gimage) < tol:
            print(f"Converged after {it+1} iterations.")
            break
    else:
        print(
            f"Reached maximum iterations ({max_iter}) without full convergence."
        )

    return Cpixel, Gimage


def estimate_gain_for_raw(raw: np.ndarray, Cpixel: np.ndarray) -> float:
    """
    Estimate the gain Gimage for a new raw file given the already-computed Cpixel.

    Parameters
    ----------
    raw : np.ndarray shape (N,)
        Raw data of the new file.
    Cpixel : np.ndarray shape (N,)
        Per-pixel correction factors.

    Returns
    -------
    Gimage : float
        Estimated gain (>0).
    """
    eps = 1e-12
    denom = np.sum((Cpixel * raw) ** 2) + eps
    numer = np.sum(Cpixel * raw) * 50000.0
    return numer / denom


def save_correction_factors_csv(Cpixel: np.ndarray, output_path: str = "correction.csv"):
    """
    Save Cpixel to a CSV file (one column).

    Parameters
    ----------
    Cpixel : np.ndarray
        Correction factors.
    output_path : str
        Path of the CSV file to write.
    """
    # Ensure Cpixel is 1-D
    Cpixel = Cpixel.ravel()
    np.savetxt(output_path, Cpixel, delimiter=",")
    print(f"Correction factors written to '{output_path}'.")


def main():
    parser = argparse.ArgumentParser(
        description="Estimate per-pixel correction factors and per-file gains."
    )
    parser.add_argument("--input", required=True, help="Folder containing raw files")
    parser.add_argument("--ext", required=True, help="file extension to process (.high or .low)")
    parser.add_argument("--csv", required=True, help="output containing correction csv")
    args = parser.parse_args()

    input_folder = args.input
    if not os.path.isdir(input_folder):
        raise NotADirectoryError(f"'{input_folder}' is not a directory.")

    print(f"Loading raw files from '{input_folder}' …")
    raw_arrays, filenames = load_raw_files(input_folder, args.ext)
    print(f"Loaded {len(raw_arrays)} files, each with {raw_arrays[0].size} pixels.")

    print("Estimating correction factors and gains …")
    Cpixel, Gimage = fit_correction_factors(raw_arrays)

    # Print Gimage values
    #print("\nEstimated gains (Gimage) for each file:")
    #for fn, gi in zip(filenames, Gimage):
    #    print(f"  {fn:20s} : {gi:.6f}")

    # Save Cpixel to CSV
    save_correction_factors_csv(Cpixel, args.csv)

    # Example of using the helper function on a new raw file
    if raw_arrays:
        example_raw = raw_arrays[0]
        gi_example = estimate_gain_for_raw(example_raw, Cpixel)
        print(f"\nEstimated Gimagemage for first file using helper: {gi_example:.6f}")


if __name__ == "__main__":
    main()
