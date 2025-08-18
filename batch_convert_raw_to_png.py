#!/usr/bin/env python3
"""
batch_convert_raw_to_png.py

Batch convert all .raw files in a directory to 16‑bit little‑endian PNGs using
the raw_to_png.py program. The image size is assumed to be 2048×2560.

Usage:
    python batch_convert_raw_to_png.py --input-dir /path/to/raw/files [--output-dir /path/to/output]

If output directory not specified, the PNGs are written in the same folder as
the raw files.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import multiprocessing

def convert_file(raw_path: Path, png_path: Path):
    """Call raw_to_png.py to convert a single file."""
    cmd = [
        sys.executable,
        "raw_to_png.py",
        str(raw_path),
        str(png_path),
        "--width", "2048",
        "--height", "2560",
        "--channels", "1",
        "--keep-16bit",
        # raw_to_png defaults to little endian, but we can enforce it:
        "--endianness", "little"
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(f"✅ {raw_path.name} → {png_path.name}")
    except subprocess.CalledProcessError as exc:
        print(f"❌ Failed to convert {raw_path.name}: {exc.stderr.decode()}", file=sys.stderr)

def run_in_pool(raw_files, output_dir):
    with multiprocessing.Pool() as pool:
        for raw_file in raw_files:
            png_file = output_dir / (raw_file.stem + ".png")
            pool.apply_async(convert_file, args=(raw_file, png_file))
        pool.close()
        pool.join()

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", "-i", type=Path, required=True,
                        help="Directory containing .raw files.")
    parser.add_argument("--output-dir", "-o", type=Path, default=None,
                        help="Optional directory to write PNGs. If omitted, uses input dir.")
    args = parser.parse_args()

    input_dir = args.input_dir.resolve()
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir.resolve() if args.output_dir else input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = list(input_dir.glob("*.raw"))
    if not raw_files:
        print(f"No .raw files found in {input_dir}.", file=sys.stderr)
        sys.exit(0)

    run_in_pool(raw_files, output_dir)

if __name__ == "__main__":
    main()