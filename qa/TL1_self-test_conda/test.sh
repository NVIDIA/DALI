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
    for DIRNAME in \
      "../../build/dali/python/nvidia/dali" \
      "$(python -c 'import os; from nvidia import dali; print(os.path.dirname(dali.__file__))' 2>/dev/null || echo '')"
    do
        if [ -x "$DIRNAME/test/$BINNAME" ]; then
            FULLPATH="$DIRNAME/test/$BINNAME"
            break
        fi
    done

    if [[ -z "$FULLPATH" ]]; then
        echo "ERROR: $BINNAME not found"
        exit 1
    fi

    # DecodeJPEGHost doesn't work well for Conda as OpenCV there uses libjpeg that returns a bit different
    # results that libturbo-jpeg DALI uses, also OpenCV conflicts with FFMpeg >= 4.2 which is reguired
    # to handle PackedBFrames
    "$FULLPATH" --gtest_filter="*:-*Vp9*"
  done
}

pushd ../..
source ./qa/test_template.sh
popd
