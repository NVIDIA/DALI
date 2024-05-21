#!/bin/bash -ex

do_once() {
  # generate file_list.txt for label testing
  echo "${DALI_EXTRA_PATH}/db/video/frame_num_timestamp/test.mp4 0 0 99
  ${DALI_EXTRA_PATH}/db/video/frame_num_timestamp/test.mp4 1 100 199
  ${DALI_EXTRA_PATH}/db/video/frame_num_timestamp/test.mp4 2 200 256
  " > /tmp/file_list.txt

  echo "${DALI_EXTRA_PATH}/db/video/frame_num_timestamp/test_25fps.mp4 0 0 1
  ${DALI_EXTRA_PATH}/db/video/frame_num_timestamp/test_25fps.mp4 1 2 3
  ${DALI_EXTRA_PATH}/db/video/frame_num_timestamp/test_25fps.mp4 2 4 5
  " > /tmp/file_list_timestamp.txt

  echo "${DALI_EXTRA_PATH}/db/video/frame_num_timestamp/test_25fps.mp4 0 0 49
  ${DALI_EXTRA_PATH}/db/video/frame_num_timestamp/test_25fps.mp4 1 50 99
  ${DALI_EXTRA_PATH}/db/video/frame_num_timestamp/test_25fps.mp4 2 100 149
  " > /tmp/file_list_frame_num.txt
}

# populate epilog and prolog with variants to enable/disable conda
# every test will be executed for bellow configs
prolog=(enable_conda)
epilog=(disable_conda)

test_body() {
  for BINNAME in \
    "dali_core_test.bin" \
    "dali_kernel_test.bin" \
    "dali_test.bin" \
    "dali_operator_test.bin"
  do
    # results that libturbo-jpeg DALI uses, also OpenCV conflicts with FFMpeg >= 4.2 which is reguired
    # to handle PackedBFrames
    # use `which` to invoke test binary with full path so
    # https://google.github.io/googletest/advanced.html#death-test-styles which runs tests in
    # a separate process don't use PATH to discover the file location and fails
    $(which $BINNAME) --gtest_filter="*:-*Vp9*"
  done
}

pushd ../..
source ./qa/test_template.sh
popd
