#!/bin/bash -e

pip_packages='${python_test_runner_package} numpy opencv-python-headless av'
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

  mkdir -p /tmp/video_resolution/
  cp -r ${DALI_EXTRA_PATH}/db/video_resolution/vp9/ /tmp/video_resolution/

  container_path=${DALI_EXTRA_PATH}/db/optical_flow/sintel_trailer/sintel_trailer_vp9.mp4

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

  # MPEG-4 Part 2 in AVI with B-frames and open GOPs, each frame showing its number as the one
  # bright cell of a 10x9 grid of 16x14 cells (see test_video_reader.MPEG4_OPEN_GOP_AVI). The
  # noise keeps every macroblock coded: skipped ones in B-VOPs are not decoded alike by all decoders.
  mkdir -p /tmp/avi_video_files
  ffmpeg -f lavfi -i "color=black:size=160x128:rate=30,format=yuv420p,geq=lum='if(eq(floor(X/16)+10*floor(Y/14)\,N)\,235\,16+60*random(1)*lt(mod(X\,8)\,4))':cb=128:cr=128" \
      -frames:v 90 -c:v mpeg4 -bf 2 -g 15 -q:v 2 -y /tmp/avi_video_files/mpeg4_open_gop.avi

  # The same frames in H.264 with B-frames and an IDR picture every 15 frames (more than one, unlike
  # the AVI files of DALI_extra; see test_video_reader.H264_IDR_GOPS_AVI).
  ffmpeg -f lavfi -i "color=black:size=160x128:rate=30,format=yuv420p,geq=lum='if(eq(floor(X/16)+10*floor(Y/14)\,N)\,235\,16+60*random(1)*lt(mod(X\,8)\,4))':cb=128:cr=128" \
      -frames:v 90 -c:v libx264 -bf 2 -g 15 -keyint_min 15 -sc_threshold 0 -pix_fmt yuv420p \
      -y /tmp/avi_video_files/h264_idr_gops.avi
  # And the same without B-frames, and with three B-frames in a pyramid (see
  # test_video_reader.H264_IDR_GOPS_VARIANT_AVIS).
  for variant in "no_bframes:-bf 0" "pyramid:-bf 3 -x264-params b-pyramid=normal"; do
      ffmpeg -f lavfi -i "color=black:size=160x128:rate=30,format=yuv420p,geq=lum='if(eq(floor(X/16)+10*floor(Y/14)\,N)\,235\,16+60*random(1)*lt(mod(X\,8)\,4))':cb=128:cr=128" \
          -frames:v 90 -c:v libx264 -g 15 -keyint_min 15 -sc_threshold 0 ${variant#*:} -pix_fmt yuv420p \
          -y /tmp/avi_video_files/h264_idr_gops_${variant%%:*}.avi
  done
}

test_body() {
    # test code
    # First running simple code
    python video_label_example.py

    echo $(pwd)
    pushd ../../../../dali/test/python/

    ${python_new_invoke_test} test_video_pipeline
    ${python_new_invoke_test} test_video_reader_resize
    ${python_new_invoke_test} test_video_reader
    ${python_new_invoke_test} test_video_reader_index
    ${python_new_invoke_test} test_video_reader_output_buffers

    popd
}

pushd ../..
source ./qa/test_template.sh
popd
