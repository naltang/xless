#!/usr/bin/env bash

mkdir output

python3 batch_convert_raw_to_png.py --input-dir=data/center --output-dir=output/center
python3 batch_convert_raw_to_png.py --input-dir=data/empty --output-dir=output/empty
python3 batch_convert_raw_to_png.py --input-dir=data/leftdown --output-dir=output/leftdown
python3 batch_convert_raw_to_png.py --input-dir=data/leftdown-longterm --output-dir=output/leftdown-longterm
python3 batch_convert_raw_to_png.py --input-dir=data/leftup --output-dir=output/leftup
python3 batch_convert_raw_to_png.py --input-dir=data/rightdown --output-dir=output/rightdown
python3 batch_convert_raw_to_png.py --input-dir=data/rightup --output-dir=output/rightup
