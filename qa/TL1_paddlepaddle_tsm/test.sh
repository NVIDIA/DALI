#!/bin/bash -e
# used pip packages
# rarfile>= 3.2 breaks python 3.5 compatibility
pip_packages='paddlepaddle-gpu'
target_dir=./docs/examples/use_cases/paddle/tsm/

do_once() {
  apt-get update
  apt-get install -y ffmpeg

  mkdir -p demo
  ffmpeg -y -i ${DALI_EXTRA_PATH}/db/video/sintel/labelled_videos/1/sintel_trailer-720p_7.mp4 \
         -filter:v scale=534:300 -ss 0 -t 10 -c:a copy demo/7.mp4

  mkdir -p ~/.cache/paddle/weights/
}

test_body() {
  out=`python infer.py -k 1 -s 15 demo`
  if echo $out | grep -E "(waxing_legs|paragliding)"; then
    exit 0
  fi
  exit 3
}

pushd ../..
source ./qa/test_template.sh
popd
