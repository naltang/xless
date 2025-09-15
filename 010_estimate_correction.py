#!/usr/bin/env python3
"""
Least-squares estimation of per-pixel correction factors Cp and per-file gains Gi
from 16-bit raw images.

Usage:
    python estimate_correction.py /path/to/raw_folder

The program will:
    - read all *.raw files in the supplied folder (they must have identical length)
    - compute Cp and Gi that minimise Σ_i,k [RAW_i[k] * Cp[k] * Gi[i] – 50000]^2
      with mean(Cp)=1
    - write Cp to correctoin.csv
    - print the Gi values for all files
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
    Perform alternating least-squares to estimate Cp and Gi.

    Parameters
    ----------
    raw_arrays : list of np.ndarray
        Raw data for each file (shape (N,))
    max_iter : int, optional
        Maximum number of ALS iterations.
    tol : float, optional
        Convergence tolerance on the maximum change in Cp or Gi.

    Returns
    -------
    Cp : np.ndarray shape (N,)
        Per-pixel correction factors (normalised so mean(Cp)=1)
    Gi : np.ndarray shape (M,)
        Per-file gain scalars (>0)
    """
    # Convert to a 2-D array for easier vectorisation: shape (M, N)
    Y = np.stack(raw_arrays)          # dtype float64
    M, N = Y.shape

    eps = 1e-12                      # avoid division by zero

    Cp = np.ones(N, dtype=np.float64)

    Gi = np.ones(M, dtype=np.float64)   # initial guess (will be updated)
    for it in range(max_iter):
        print(f"iteration {it}")
        old_Cp = Cp.copy()
        old_Gi = Gi.copy()

        # ---------- Update Gi ----------
        # For each file i: minimise Σ_k [Cp[k]*Gi[i]*Y[i,k] – 50000]^2
        denom_Gi = np.sum((Cp * Y)**2, axis=1) + eps          # shape (M,)
        numer_Gi = np.sum(Cp * Y, axis=1) * 50000.0           # shape (M,)
        Gi = numer_Gi / denom_Gi

        # ---------- Update Cp ----------
        # For each pixel k: minimise Σ_i [Cp[k]*Gi[i]*Y[i,k] – 50000]^2
        # Vectorised over all pixels:
        Gi_col = Gi[:, None]                                 # shape (M,1)
        denom_Cp = np.sum((Gi_col * Y)**2, axis=0) + eps     # shape (N,)
        numer_Cp = np.sum(Gi_col * Y, axis=0) * 50000.0      # shape (N,)
        Cp = numer_Cp / denom_Cp

        # Normalise Cp so that its mean is 1
        Cp /= Cp.mean()

        # ---------- Convergence check ----------
        delta_Cp = np.max(np.abs(Cp - old_Cp))
        delta_Gi = np.max(np.abs(Gi - old_Gi))
        print(f"Cp= ... {Cp[N//2 + 1000 : N//2 + 1005]} ...")
        print(f"Gi= ... {Gi[M//2 : M//2 + 5]} ...")
        print(f"tol={tol}, delta_Cp={delta_Cp}, delta_Gi={delta_Gi}")
        if max(delta_Cp, delta_Gi) < tol:
            print(f"Converged after {it+1} iterations.")
            break
    else:
        print(
            f"Reached maximum iterations ({max_iter}) without full convergence."
        )

    return Cp, Gi


def estimate_gain_for_raw(raw: np.ndarray, Cp: np.ndarray) -> float:
    """
    Estimate the gain Gi for a new raw file given the already-computed Cp.

    Parameters
    ----------
    raw : np.ndarray shape (N,)
        Raw data of the new file.
    Cp : np.ndarray shape (N,)
        Per-pixel correction factors.

    Returns
    -------
    Gi : float
        Estimated gain (>0).
    """
    eps = 1e-12
    denom = np.sum((Cp * raw) ** 2) + eps
    numer = np.sum(Cp * raw) * 50000.0
    return numer / denom


def save_correction_factors_csv(Cp: np.ndarray, output_path: str = "correctoin.csv"):
    """
    Save Cp to a CSV file (one column).

    Parameters
    ----------
    Cp : np.ndarray
        Correction factors.
    output_path : str
        Path of the CSV file to write.
    """
    # Ensure Cp is 1-D
    Cp = Cp.ravel()
    np.savetxt(output_path, Cp, delimiter=",")
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
    Cp, Gi = fit_correction_factors(raw_arrays)

    # Print Gi values
    #print("\nEstimated gains (Gi) for each file:")
    #for fn, gi in zip(filenames, Gi):
    #    print(f"  {fn:20s} : {gi:.6f}")

    # Save Cp to CSV
    save_correction_factors_csv(Cp, args.csv)

    # Example of using the helper function on a new raw file
    if raw_arrays:
        example_raw = raw_arrays[0]
        gi_example = estimate_gain_for_raw(example_raw, Cp)
        print(f"\nEstimated Gi for first file using helper: {gi_example:.6f}")


if __name__ == "__main__":
    main()
