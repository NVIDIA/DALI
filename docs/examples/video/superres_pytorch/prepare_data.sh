#!/bin/bash

set -e

if [ -z "$1" ]
then
  echo "Usage ./prepare_data.sh [FILE]"
  exit 1
fi

python ./tools/split_scenes.py --raw_data $1 --out_data data_dir

python ./tools/transcode_scenes.py --master_data data_dir --resolution 540p
python ./tools/transcode_scenes.py --master_data data_dir --resolution 720p
python ./tools/transcode_scenes.py --master_data data_dir --resolution 1080p
python ./tools/transcode_scenes.py --master_data data_dir --resolution 4K
