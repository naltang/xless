#!/usr/bin/env bash
python3 000_batch_median_crop.py --input data/empty --output output/empty
python3 000_batch_median_crop.py --input data/empty/on --output output/empty/on

python3 000_batch_median_crop.py --input data/center --output output/center
python3 000_batch_median_crop.py --input data/leftup --output output/leftup
python3 000_batch_median_crop.py --input data/leftdown --output output/leftdown
python3 000_batch_median_crop.py --input data/rightup --output output/rightup
python3 000_batch_median_crop.py --input data/rightdown --output output/rightdown

python3 000_batch_median_crop.py --input data/center/on --output output/center/on
python3 000_batch_median_crop.py --input data/leftup/on --output output/leftup/on
python3 000_batch_median_crop.py --input data/leftdown/on --output output/leftdown/on
python3 000_batch_median_crop.py --input data/rightup/on --output output/rightup/on
python3 000_batch_median_crop.py --input data/rightdown/on --output output/rightdown/on

python3 010_estimate_correction.py --input output/empty --ext low --csv output/correction_low.csv
python3 010_estimate_correction.py --input output/empty --ext high --csv output/correction_high.csv


python3 020_apply_correction.py --input output/empty --output output/empty --high output/correction_high.csv --low output/correction_low.csv
python3 020_apply_correction.py --input output/center --output output/center --high output/correction_high.csv --low output/correction_low.csv
python3 020_apply_correction.py --input output/leftup --output output/leftup --high output/correction_high.csv --low output/correction_low.csv
python3 020_apply_correction.py --input output/leftdown --output output/leftdown --high output/correction_high.csv --low output/correction_low.csv
python3 020_apply_correction.py --input output/rightup --output output/rightup --high output/correction_high.csv --low output/correction_low.csv
python3 020_apply_correction.py --input output/rightdown --output output/rightdown --high output/correction_high.csv --low output/correction_low.csv
