#!/usr/bin/env python3
"""
Convert a raw file containing unsigned 16‑bit integers into a PNG image.

Typical usage:

    python3 raw_to_png.py input.raw output.png \
        --width 1024 --height 768

Optional arguments:
    --channels   Number of interleaved channels (1 for grayscale, 3 for RGB).
                 Default: 1
    --keep-16bit Preserve the 16‑bit depth in the PNG.
    --endianness little|big   Endianness of the raw data. Default is little.

Author: OpenAI ChatGPT
"""

import argparse
import sys

import numpy as np
from PIL import Image


def _dtype_for_endianness(endianness: str):
    """Return a NumPy dtype object for unsigned 16‑bit integers with the given endianness."""
    if endianness == "little":
        return np.dtype("<u2")   # little‑endian unsigned short
    elif endianness == "big":
        return np.dtype(">u2")   # big‑endian unsigned short
    else:
        raise ValueError("endianness must be 'little' or 'big'")


def raw_to_png(
    input_path: str,
    output_path: str,
    width: int,
    height: int,
    channels: int = 1,
    keep_16bit: bool = False,
    endianness: str = "little",
):
    """
    Convert a raw file into a PNG.

    Parameters
    ----------
    input_path : str
        Path to the binary raw file.
    output_path : str
        Path where the PNG will be written.
    width, height : int
        Image dimensions. The script expects exactly `width*height*channels*2` bytes.
    channels : int, optional
        Number of interleaved channels (1=grayscale, 3=RGB). Default is 1.
    keep_16bit : bool, optional
        If True, the PNG will retain the original 16‑bit depth. Otherwise,
        data are linearly scaled to 8‑bit per channel.
    endianness : str, optional
        Endianness of the raw data: "little" (default) or "big".
    """
    dtype = _dtype_for_endianness(endianness)

    # ------------------------------------------------------------------
    # 1. Read the binary file into a NumPy array
    # ------------------------------------------------------------------
    try:
        with open(input_path, "rb") as fh:
            raw_bytes = fh.read()
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Input file not found: {input_path}") from exc

    expected_size = width * height * channels * 2
    if len(raw_bytes) != expected_size:
        raise ValueError(
            f"File size mismatch:\n"
            f"  Expected: {expected_size} bytes (for {width}×{height}×{channels})\n"
            f"  Actual:   {len(raw_bytes)} bytes\n"
            "Either the dimensions are wrong or the file is corrupted."
        )

    # Interpret the byte stream as a flat array of unsigned 16‑bit integers
    data = np.frombuffer(raw_bytes, dtype=dtype)

    # Reshape to (height, width) for grayscale or (height, width, channels)
    try:
        if channels == 1:
            data = data.reshape((height, width))
        else:
            data = data.reshape((height, width, channels))
    except ValueError as exc:
        raise ValueError("Failed to reshape the raw data into the requested dimensions.") from exc

    # ------------------------------------------------------------------
    # 2. Convert / scale to 8‑bit (if requested)
    # ------------------------------------------------------------------
    if keep_16bit:
        # Pillow can write 16‑bit PNGs using mode 'I;16' or 'RGB;16'.
        # We'll use the appropriate mode based on channel count.
        if channels == 1:
            img = Image.fromarray(data, mode="I;16")
        elif channels == 3:
            img = Image.fromarray(data, mode="RGB;16")
        else:
            raise ValueError("PNG 16‑bit support is implemented only for 1 or 3 channels.")
    else:
        # Scale the data to 0–255.  We use the global min/max of the array.
        min_val = int(np.min(data))
        max_val = int(np.max(data))

        if max_val == min_val:
            # Avoid division by zero – all pixels are identical
            scaled = np.zeros_like(data, dtype=np.uint8)
        else:
            scale_factor = 255.0 / (max_val - min_val)
            scaled = ((data.astype(np.float32) - min_val) * scale_factor).clip(0, 255).astype(np.uint8)

        if channels == 1:
            img = Image.fromarray(scaled, mode="L")   # 8‑bit grayscale
        elif channels == 3:
            img = Image.fromarray(scaled, mode="RGB") # 8‑bit RGB
        else:
            raise ValueError("PNG 8‑bit support is implemented only for 1 or 3 channels.")

    # ------------------------------------------------------------------
    # 3. Write the PNG file
    # ------------------------------------------------------------------
    try:
        img.save(output_path, format="PNG")
    except Exception as exc:
        raise RuntimeError(f"Failed to write PNG: {output_path}") from exc


def _parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("input", help="Path to the raw input file")
    parser.add_argument("output", help="Path for the output PNG")

    parser.add_argument("--width", type=int, required=True, help="Image width in pixels")
    parser.add_argument("--height", type=int, required=True, help="Image height in pixels")
    parser.add_argument(
        "--channels",
        type=int,
        default=1,
        choices=[1, 3],
        help="Number of interleaved channels (default: 1)",
    )
    parser.add_argument(
        "--keep-16bit",
        action="store_true",
        help="Keep the original 16‑bit depth in the PNG (no scaling).",
    )
    parser.add_argument(
        "--endianness",
        choices=["little", "big"],
        default="little",
        help="Byte order of the raw data; 'little' is typical on x86. Default: little",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    try:
        raw_to_png(
            input_path=args.input,
            output_path=args.output,
            width=args.width,
            height=args.height,
            channels=args.channels,
            keep_16bit=args.keep_16bit,
            endianness=args.endianness,
        )
        print(f"✅ Successfully converted {args.input} → {args.output}")
    except Exception as exc:
        print(f"\n❌ Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
