#!/usr/bin/env bash
rm -f run.log
python3 000_batch_median_crop.py --in data/empty --out outpyt/empty | tee -a run.log
python3 010_estimate_correction.py --in output/empty --ext low --csv output/correction_low.csv | tee -a run.log
python3 010_estimate_correction.py --in output/empty --ext high --csv output/correction_high.csv | tee -a run.log
python3 020_apply_correction.py --in output/empty --out output/empty --high output/correction_high.csv --low output/correction_low.csv