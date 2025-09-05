#!/usr/bin/env bash
rm -f run.log
python3 000_batch_median_crop.py data/empty outpyt/empty | tee -a run.log
python3 010_estimate_correction.py --folder output/empty --ext low --csv correction_low.csv | tee -a run.log
python3 010_estimate_correction.py --folder output/empty --ext high --csv correction_high.csv | tee -a run.log
python3 020_apply_correction.py 