#!/bin/bash -e

pip_packages='${python_test_runner_package} numpy opencv-python-headless'
target_dir=./docs/examples/sequence_processing/video

do_once() {
  apt-get update
  apt-get install -y wget ffmpeg

  TMP_VIDEO_FILES=/tmp/video_files
  TMP_MANY_VIDEO_FILES=/tmp/many_video_files
  TMP_LABLED_VIDEO_FILES=/tmp/labelled_videos

  mkdir -p $TMP_VIDEO_FILES
  mkdir -p $TMP_MANY_VIDEO_FILES
  mkdir -p $TMP_LABLED_VIDEO_FILES/{0..2}
  cp -r ${DALI_EXTRA_PATH}/db/video_resolution /tmp/

  container_path=${DALI_EXTRA_PATH}/db/optical_flow/sintel_trailer/sintel_trailer.mp4

  IFS='/' read -a container_name <<< "$container_path"
  IFS='.' read -a split <<< "${container_name[-1]}"

  for i in {0..4};
  do
      ffmpeg -ss 00:00:${i}0 -t 00:00:10 -i $container_path -vcodec copy -acodec copy -y $TMP_VIDEO_FILES/${split[0]}_$i.${split[1]}
  done

  #create dummy files for max_opened_file_test
  for i in {0..1400};
  do
      ln -s ../video_files/${split[0]}_0.${split[1]} $TMP_MANY_VIDEO_FILES/${split[0]}_$i.${split[1]}
  done

  for i in {0..9};
  do
      ffmpeg -ss 00:00:$((i*5)) -t 00:00:05 -i $container_path -vcodec copy -acodec copy -y $TMP_LABLED_VIDEO_FILES/$((i % 3))/${split[0]}_$i.${split[1]}
  done

  # generate file_list.txt from video_files directory
  ls -d $TMP_VIDEO_FILES/*  | tr " " "\n" | awk '{print $0, NR;}' > /tmp/file_list.txt
}

test_body() {
    # test code
    # First running simple code
    python video_label_example.py

    echo $(pwd)
    ${python_invoke_test} ../../../../dali/test/python/test_video_pipeline.py
    ${python_invoke_test} ../../../../dali/test/python/test_video_reader_resize.py

    cd ../../../../dali/test/python/
    ${python_new_invoke_test} test_video_reader
}

pushd ../..
source ./qa/test_template.sh
popd
