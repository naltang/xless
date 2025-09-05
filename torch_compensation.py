#!/usr/bin/env python3
"""
estimate_rr_gg.py

Given a folder that contains many 16‑bit raw gray images (2560×2048),
this script estimates:
    * rr[j] – a pixel‑wise recovery factor   (shape H × W)
    * gg[i] – a scalar gain for image i      (positive)

The model is

        pv_ij · rr_j · gg_i  ≈  TARGET          (TARGET = 50 000)

rr is common to all images, gg is unique per image.
The algorithm alternates between updating the gains and the recovery
matrix until convergence.

The script:
    • splits the data into a training and a validation set,
      trains rr on the training set,
      prints/returns the gains for every image,
      writes rr to ``rr.csv`` (H×W),
      and offers a helper function that can compute gg for a new raw image.
"""

import argparse
import glob
import os
from pathlib import Path

import numpy as np
import torch


# --------------------------------------------------------------------------- #
# 1. Helper functions – loading, splitting, training, evaluation
# --------------------------------------------------------------------------- #

def load_raw_images(folder: str) -> list[tuple[str, torch.Tensor]]:
    """
    Load all *.raw files from *folder* into a list of (filename, tensor).

    Each raw file is read as uint16 and cast to float32.
    The returned tensors are moved to the chosen device.

    Returns
    -------
    images : List[(str, Tensor)]
        A list where each element contains the filename (without path)
        and a 2‑D tensor of shape (H, W).
    """
    raw_paths = sorted(glob.glob(os.path.join(folder, "*.raw")))
    if not raw_paths:
        raise FileNotFoundError(f"No *.raw files found in {folder}")

    images = []
    for p in raw_paths:
        # Raw is stored as little‑endian unsigned 16‑bit
        data = np.fromfile(p, dtype="<u2")
        if data.size != 2560 * 2048:
            raise ValueError(f"File {p} has unexpected size {data.size}")
        img = torch.from_numpy(data.astype(np.float32)).reshape(2560, 2048)
        images.append((Path(p).stem, img.to(device)))
    return images


def train_val_split(images: list[tuple[str, torch.Tensor]],
                    val_ratio: float = 0.2,
                    seed: int | None = None) -> tuple[
                        list[tuple[str, torch.Tensor]],
                        list[tuple[str, torch.Tensor]]]:
    """
    Randomly split *images* into training and validation sets.

    Parameters
    ----------
    images : List[(str, Tensor)]
        All loaded images.
    val_ratio : float
        Fraction of data used for validation (default 0.2).
    seed : int | None
        Optional random seed for reproducibility.

    Returns
    -------
    train_imgs, val_imgs : tuple[List, List]
    """
    if seed is not None:
        np.random.seed(seed)

    idx = np.arange(len(images))
    np.random.shuffle(idx)
    n_val = max(1, int(val_ratio * len(images)))
    val_idx, train_idx = idx[:n_val], idx[n_val:]

    return [images[i] for i in train_idx], [images[i] for i in val_idx]


def estimate_rr_gg(train_imgs: list[tuple[str, torch.Tensor]],
                   target: float = 50000.0,
                   max_iter: int = 200,
                   tol: float = 1e-4) -> tuple[torch.Tensor, dict]:
    """
    Estimate the recovery matrix rr and per‑image gains gg on *train_imgs*.

    The algorithm alternates:
        1. For every image i, compute
              gg_i = target / mean(raw_ij · rr_j)
        2. For every pixel j, compute
              rr_j = mean_{i} [ target / (raw_ij · gg_i) ]

    Iteration stops when the relative change in the loss falls below *tol*
    or when *max_iter* iterations have been performed.

    Parameters
    ----------
    train_imgs : List[(str, Tensor)]
        Training images.
    target : float
        Desired constant value (50 000).
    max_iter : int
        Maximum number of alternating updates.
    tol : float
        Relative tolerance for stopping criterion.

    Returns
    -------
    rr : Tensor (H × W)
        Pixel‑wise recovery matrix, positive values.
    gg_dict : Dict[str, float]
        Per‑image gain values indexed by filename.
    """
    # Initialise rr as ones – any positive value works
    H, W = 2560, 2048
    rr = torch.ones(H, W, device=device, dtype=torch.float32)

    # Pre‑allocate tensors for efficient computation
    raw_stack = torch.stack([img for _, img in train_imgs], dim=0)   # (N, H, W)
    names = [name for name, _ in train_imgs]
    N_train = raw_stack.shape[0]

    # Add a small epsilon to avoid division by zero
    eps = 1e-8

    prev_loss = float("inf")
    for it in range(1, max_iter + 1):
        # ---------- Update gains ----------
        # mean over pixels: shape (N,)
        mean_raw_rr = torch.mean(raw_stack * rr[None, :, :], dim=(1, 2))
        gg = target / (mean_raw_rr + eps)                     # shape (N,)

        # ---------- Update recovery matrix ----------
        # For each pixel j: average over images
        inv_raw_gg = target / (raw_stack * gg[:, None, None] + eps)
        rr_new = torch.mean(inv_raw_gg, dim=0)                # shape (H,W)

        # ---------- Loss ----------
        pred = raw_stack * rr_new[None, :, :] * gg[:, None, None]
        loss = torch.mean((pred - target) ** 2).item()

        rel_change = abs(prev_loss - loss) / max(1.0, prev_loss)
        if it % 10 == 0 or it == 1:
            print(f"Iter {it:3d} | Loss={loss:.4f} | Δ={rel_change:.6f}")
        print(f"Iter {it:3d} | Loss={loss:.4f} | Δ={rel_change:.6f}")

        if rel_change < tol and it > 5:   # ignore early iterations
            rr = rr_new
            break

        rr = rr_new
        prev_loss = loss

    # Build dictionary of gains (float values)
    gg_dict = {name: float(g) for name, g in zip(names, gg.cpu().numpy())}
    return rr.detach(), gg_dict


def evaluate_validation(rr: torch.Tensor,
                        val_imgs: list[tuple[str, torch.Tensor]],
                        target: float = 50000.0) -> float:
    """
    Compute mean squared error on the validation set.

    Parameters
    ----------
    rr : Tensor (H × W)
        Trained recovery matrix.
    val_imgs : List[(str, Tensor)]
        Validation images.
    target : float
        Desired constant value.

    Returns
    -------
    mse : float
        Mean squared error over all pixels in the validation set.
    """
    raw_stack = torch.stack([img for _, img in val_imgs], dim=0)   # (N_val, H, W)
    N_val = raw_stack.shape[0]

    # Estimate gains for validation images using current rr
    mean_raw_rr = torch.mean(raw_stack * rr[None, :, :], dim=(1, 2))
    gg_val = target / (mean_raw_rr + 1e-8)

    pred = raw_stack * rr[None, :, :] * gg_val[:, None, None]
    mse = torch.mean((pred - target) ** 2).item()
    return mse


def get_gg_for_new(raw: torch.Tensor,
                   rr: torch.Tensor,
                   target: float = 50000.0) -> float:
    """
    Compute the gain for a *new* raw image using the learned recovery matrix.

    Parameters
    ----------
    raw : Tensor (H × W)
        New raw image, dtype=float32.
    rr : Tensor (H × W)
        Learned pixel‑wise recovery matrix.
    target : float
        Desired constant value.

    Returns
    -------
    gg_new : float
        Estimated scalar gain for the new image.
    """
    mean_raw_rr = torch.mean(raw * rr).item()
    return float(target / (mean_raw_rr + 1e-8))


# --------------------------------------------------------------------------- #
# 2. Main routine – command line interface
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Estimate rr and gg from raw images.")
    parser.add_argument("folder", type=str, help="Folder containing .raw files")
    parser.add_argument("--val_ratio", type=float, default=0.2,
                        help="Fraction of data used for validation (default 0.2)")
    parser.add_argument("--max_iter", type=int, default=200,
                        help="Maximum number of iterations for alternating updates")
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="Relative tolerance to stop the algorithm")
    parser.add_argument("--seed", type=int, default=None,
                        help="Random seed for train/val split (default None)")
    args = parser.parse_args()

    print("Loading raw images …")
    images = load_raw_images(args.folder)
    print(f"  {len(images)} images loaded.")

    # Split into training and validation
    train_imgs, val_imgs = train_val_split(images,
                                           val_ratio=args.val_ratio,
                                           seed=args.seed)

    print(f"Training set: {len(train_imgs)} images")
    print(f"Validation set: {len(val_imgs)} images")

    # Train rr and compute gains for training images
    print("\nEstimating recovery matrix (rr) …")
    rr, gg_dict = estimate_rr_gg(train_imgs,
                                 target=50000.0,
                                 max_iter=args.max_iter,
                                 tol=args.tol)

    # Print gains for all images (train + validation)
    print("\nPer‑image gains:")
    for name in sorted(gg_dict):
        print(f"  {name:20s} : gg = {gg_dict[name]:.6f}")

    # Evaluate on validation set
    val_mse = evaluate_validation(rr, val_imgs, target=50000.0)
    print(f"\nValidation MSE (mean squared error): {val_mse:.2f}")

    # Save rr to CSV
    csv_path = Path(args.folder) / "rr.csv"
    np.savetxt(csv_path, rr.cpu().numpy(), delimiter=",")
    print(f"Recovery matrix saved to {csv_path}")

    # Demonstrate helper for a new image (optional)
    if val_imgs:
        raw_new = val_imgs[0][1]  # just reuse the first validation image
        gg_new = get_gg_for_new(raw_new, rr, target=50000.0)
        print(f"\nExample: computed gg for '{val_imgs[0][0]}' using rr -> {gg_new:.6f}")


if __name__ == "__main__":
    # Detect GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}\n")
    main()
