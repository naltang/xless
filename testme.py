#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Estimate element-wise correction curves (COR_L, COR_H) and per-file scalar gains
(GAIN_L[i], GAIN_H[i]) for pairs of 16-bit raw grayscale arrays.

The optimisation problem is:
    low[i,k] * COR_L[k] * GAIN_L[i]   = 50000 + residual_low[i,k]
    high[i,k] * COR_H[k] * GAIN_H[i]  = 50000 + residual_high[i,k]

    residual_low / residual_high = 1 + residual_ratio

We minimise the sum of squared residuals (including the ratio term) with
PyTorch, then normalise COR_* so that their average value equals 1.

Author:  ChatGPT (2025-09-14)
"""

import os
import argparse
import glob
import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm import tqdm

# ----------------------------------------------------------------------
#  Helper functions for I/O
# ----------------------------------------------------------------------
def read_raw_uint16(path: str) -> np.ndarray:
    """
    Read a raw binary file that contains only uint16 values.
    The length of the file is inferred automatically.
    """
    data = np.fromfile(path, dtype=np.uint16)
    return data


def load_dataset(directory: str):
    """
    Scan *directory* for files named   <i>.raw.low   and   <i>.raw.high
    (i = 1 … n).  Return a list of low-arrays, a list of high-arrays and the
    integer identifiers (i).
    """
    low_paths = sorted(glob.glob(os.path.join(directory, "*.raw.low")))
    high_paths = sorted(glob.glob(os.path.join(directory, "*.raw.high")))

    # sanity check - the two lists must have the same length and matching ids
    assert len(low_paths) == len(high_paths), \
        "Unequal number of low/high files."

    ids = []
    lows = []
    highs = []

    for low_path, high_path in zip(low_paths, high_paths):
        # extract the integer id (everything before the first dot)
        basename = os.path.basename(low_path)
        i_str = basename.split(".")[0]
        i = int(i_str)

        # ensure the counterpart exists
        assert os.path.basename(high_path).startswith(i_str + "."), \
            f"File pair mismatch for id {i}"

        low_arr = read_raw_uint16(low_path)
        high_arr = read_raw_uint16(high_path)

        assert low_arr.shape == high_arr.shape, \
            f"Shape mismatch for id {i}: low {low_arr.shape}, high {high_arr.shape}"

        ids.append(i)
        lows.append(low_arr.astype(np.float32))   # cast to float for torch
        highs.append(high_arr.astype(np.float32))

    # all arrays must have the same length L
    L = lows[0].shape[0]
    for arr in lows + highs:
        assert arr.shape[0] == L, "All raw arrays must share the same length L."

    return ids, np.stack(lows), np.stack(highs)   # shape: (n, L)


# ----------------------------------------------------------------------
#  PyTorch optimisation model
# ----------------------------------------------------------------------
class CorrectionModel(nn.Module):
    """
    Learnable parameters:
        COR_L, COR_H  - shape (L,)   (global element-wise correction)
        GAIN_L, GAIN_H - shape (n,)  (per-file scalar gains)
    """
    def __init__(self, n_files: int, L: int, init_gain: float = 1.0):
        super().__init__()
        # start from 1.0 so that the normalisation later is mild
        self.COR_L = nn.Parameter(torch.ones(L, dtype=torch.float32))
        self.COR_H = nn.Parameter(torch.ones(L, dtype=torch.float32))

        # per-file gains - initialise close to 1.0 as a sensible guess
        self.GAIN_L = nn.Parameter(torch.full((n_files,), init_gain,
                                              dtype=torch.float32))
        self.GAIN_H = nn.Parameter(torch.full((n_files,), init_gain,
                                              dtype=torch.float32))

    def forward(self, low: torch.Tensor, high: torch.Tensor):
        """
        low, high : tensors of shape (n, L)
        Returns a dict with residuals and the scalar predictions.
        """
        # broadcasting: (n,1) * (1,L) → (n,L)
        pred_low  = low  * self.COR_L * self.GAIN_L.unsqueeze(1)
        pred_high = high * self.COR_H * self.GAIN_H.unsqueeze(1)

        residual_low   = pred_low   - 50000.0
        residual_high  = pred_high  - 50000.0

        # avoid division by zero - add a tiny epsilon
        eps = 1e-6
        residual_ratio = residual_low / (residual_high + eps) - 1.0

        return {
            "r_low": residual_low,
            "r_high": residual_high,
            "r_ratio": residual_ratio,
        }


def train_corrections(low_np: np.ndarray,
                      high_np: np.ndarray,
                      n_epochs: int = 5000,
                      lr: float = 1e-2,
                      lambda_ratio: float = 10.0,
                      verbose: bool = True):
    """
    Run the optimisation loop.
    Arguments:
        low_np, high_np - numpy arrays of shape (n, L) (float32)
        n_epochs        - number of optimisation steps
        lr              - learning rate for Adam
        lambda_ratio    - weight for the residual-ratio term in the loss
    Returns:
        model (trained), history (list of loss values)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    low  = torch.from_numpy(low_np).to(device)
    high = torch.from_numpy(high_np).to(device)

    n_files, L = low.shape
    model = CorrectionModel(n_files=n_files, L=L).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    iterator = range(n_epochs)
    if verbose:
        iterator = tqdm(iterator, desc="Optimising")

    for epoch in iterator:
        optimizer.zero_grad()
        out = model(low, high)

        loss_low   = (out["r_low"] ** 2).mean()
        loss_high  = (out["r_high"] ** 2).mean()
        loss_ratio = (out["r_ratio"] ** 2).mean()

        loss = loss_low + loss_high + lambda_ratio * loss_ratio
        loss.backward()
        optimizer.step()

        # Normalise the correction curves after each step (mean = 1)
        with torch.no_grad():
            model.COR_L.mul_(1.0 / model.COR_L.mean())
            model.COR_H.mul_(1.0 / model.COR_H.mean())

        loss_history.append(loss.item())

        if verbose and (epoch % 500 == 0 or epoch == n_epochs - 1):
            tqdm.write(
                f"Epoch {epoch:4d} | loss {loss.item():.3e} "
                f"(low {loss_low.item():.3e}, high {loss_high.item():.3e}, "
                f"ratio {loss_ratio.item():.3e})"
            )

    return model, loss_history


# ----------------------------------------------------------------------
#  Post-processing utilities
# ----------------------------------------------------------------------
def save_corrections_to_csv(cor_l: np.ndarray, cor_h: np.ndarray,
                            out_dir: str):
    """
    Save the two correction vectors as CSV files with a single column.
    """
    os.makedirs(out_dir, exist_ok=True)
    pd.Series(cor_l).to_csv(os.path.join(out_dir, "COR_L.csv"),
                            index=False, header=False)
    pd.Series(cor_h).to_csv(os.path.join(out_dir, "COR_H.csv"),
                            index=False, header=False)


def print_gains(ids, gain_l, gain_h):
    """
    Nicely formatted printing of the per-file scalar gains.
    """
    print("\n=== Estimated per-file gains ===")
    print(f"{'ID':>6} | {'GAIN_L':>12} | {'GAIN_H':>12}")
    print("-" * 36)
    for i, gl, gh in zip(ids, gain_l, gain_h):
        print(f"{i:6d} | {gl:12.6f} | {gh:12.6f}")


def estimate_gains(low_arr: np.ndarray,
                   high_arr: np.ndarray,
                   cor_l: np.ndarray,
                   cor_h: np.ndarray,
                   eps: float = 1e-6) -> tuple[float, float]:
    """
    Given a *new* low/high raw pair and already-known correction curves,
    compute the scalar gains that best fit the model in a least-squares sense.
    The simple closed-form solution comes from solving

        GAIN_L = mean( 50000 / (low * COR_L) )
        GAIN_H = mean( 50000 / (high * COR_H) )

    (If you also want to minimise the ratio term you could run a tiny
    local optimisation - here we keep it simple.)

    Returns:
        (gain_l, gain_h)
    """
    low = low_arr.astype(np.float32)
    high = high_arr.astype(np.float32)

    # Guard against zeros in the denominator
    denom_l = low * cor_l
    denom_h = high * cor_h
    denom_l[denom_l == 0] = eps
    denom_h[denom_h == 0] = eps

    gain_l = np.mean(50000.0 / denom_l)
    gain_h = np.mean(50000.0 / denom_h)

    return float(gain_l), float(gain_h)


# ----------------------------------------------------------------------
#  Command line interface
# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Estimate global correction curves and per-file gains "
                    "for pairs of 16-bit raw grayscale arrays."
    )
    parser.add_argument(
        "datadir",
        help="Directory that contains files <i>.raw.low and <i>.raw.high"
    )
    parser.add_argument(
        "--outdir",
        default="corrections",
        help="Folder where COR_L.csv and COR_H.csv will be written"
    )
    parser.add_argument(
        "--epochs", type=int, default=5000,
        help="Number of optimisation epochs"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-2,
        help="Learning rate for Adam optimizer"
    )
    parser.add_argument(
        "--lambda_ratio", type=float, default=10.0,
        help="Weight of the residual-ratio term in the loss"
    )
    parser.add_argument(
        "--no_verbose", action="store_true",
        help="Suppress tqdm progress bar and intermediate prints"
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------
    #  Load data
    # ------------------------------------------------------------------
    ids, low_np, high_np = load_dataset(args.datadir)
    print(f"Loaded {len(ids)} file pairs, each of length {low_np.shape[1]}")

    # ------------------------------------------------------------------
    #  Optimise
    # ------------------------------------------------------------------
    model, loss_hist = train_corrections(
        low_np, high_np,
        n_epochs=args.epochs,
        lr=args.lr,
        lambda_ratio=args.lambda_ratio,
        verbose=not args.no_verbose
    )

    # ------------------------------------------------------------------
    #  Extract results
    # ------------------------------------------------------------------
    cor_l = model.COR_L.detach().cpu().numpy()
    cor_h = model.COR_H.detach().cpu().numpy()
    gain_l = model.GAIN_L.detach().cpu().numpy()
    gain_h = model.GAIN_H.detach().cpu().numpy()

    # Normalise once more (numerical drift)
    cor_l /= cor_l.mean()
    cor_h /= cor_h.mean()

    # ------------------------------------------------------------------
    #  Save & print
    # ------------------------------------------------------------------
    save_corrections_to_csv(cor_l, cor_h, args.outdir)
    print(f"\nSaved correction curves to '{args.outdir}'")
    print_gains(ids, gain_l, gain_h)

    # ------------------------------------------------------------------
    #  Example usage of the estimator for a future raw pair
    # ------------------------------------------------------------------
    # (uncomment to see it in action)
    # new_low  = read_raw_uint16("path/to/new.raw.low").astype(np.float32)
    # new_high = read_raw_uint16("path/to/new.raw.high").astype(np.float32)
    # gL, gH = estimate_gains(new_low, new_high, cor_l, cor_h)
    # print(f"\nEstimated gains for a new pair: GAIN_L={gL:.6f}, GAIN_H={gH:.6f}")

if __name__ == "__main__":
    main()
