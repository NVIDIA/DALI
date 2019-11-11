#!/bin/bash -e
# used pip packages
pip_packages="numpy paddle"
target_dir=./docs/examples/paddle/tsm/

do_once() {
  apt-get update
  apt-get install -y youtube-dl ffmpeg

  mkdir -p demo
  youtube-dl --quiet --no-warnings -f mp4 -o demo/tmp.mp4 'https://www.youtube.com/watch?v=iU3ByohkPaM'
  ffmpeg -y -i demo/tmp.mp4 -filter:v scale=-1:300 -ss 0 -t 10 -c:a copy demo/1.mp4
  rm demo/tmp.mp4
}

test_body() {
  if python infer.py -k 1 -s 30 demo | grep 'carving_pumpkin'; then
    exit 0
  fi
  exit 3
}

pushd ../..
source ./qa/test_template.sh
popd
