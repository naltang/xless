#!/usr/bin/env bash
python3 000_batch_median_crop.py --input data/empty --output output/empty
python3 000_batch_median_crop.py --input data/empty/on --output output/empty/on
python3 010_estimate_correction.py --input output/empty --ext low --csv output/correction_low.csv
python3 010_estimate_correction.py --input output/empty --ext high --csv output/correction_high.csv
python3 020_apply_correction.py --input output/empty --output output/empty --high output/correction_high.csv --low output/correction_low.csv