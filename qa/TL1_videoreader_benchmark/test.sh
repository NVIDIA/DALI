#!/bin/bash -e

one_config_only=true
target_dir=./qa/TL1_videoreader_benchmark/

do_once() {
  apt-get update
  apt-get install -y ffmpeg linux-tools-$(uname -r)

  cp -r "${DALI_EXTRA_PATH}"/db/video_resolution .

  BENCHMARK_BIN=dali_benchmark.bin
  for DIRNAME in \
    "../../build/dali/python/nvidia/dali" \
    "$(python -c 'import os; from nvidia import dali; print(os.path.dirname(dali.__file__))' 2>/dev/null || echo '')"
  do
      if [ -x "$DIRNAME/test/$BINNAME" ]; then
          FULL_PATH="$DIRNAME/test/$BENCHMARK_BIN"
          break
      fi
  done

  if [[ -z "$FULL_PATH" ]]; then
      echo "ERROR: $BENCHMARK_BIN not found"
      exit 1
  fi

  CSV_LOG_FILE="log.csv"
  TESTCASES="testcases.csv"
  [ ! -f $TESTCASES ] && { echo "$TESTCASES file not found"; exit 1; }

  echo "Logging benchmark results to ${CSV_LOG_FILE}"
  echo "DALI(FPS)," > "$CSV_LOG_FILE"

  export DALI_TEST_VIDEO_READER_BATCH_SIZE=1;
  export DALI_TEST_VIDEO_READER_STEP=-1;
  export DALI_TEST_VIDEO_READER_TYPE=filenames;
  export DALI_TEST_VIDEO_READER_NUM_THREADS=4
  export DALI_TEST_VIDEO_READER_INITIAL_FILL=5

  #cpu_bench_frequency="2000MHz"

  while IFS=',' read -r c1 c2 c3 c4 c5 c6 c7
  do
    videos+=("$c1")
    codecs+=("$c2")
    gops+=("$c3")
    in_pixfmts+=("$c4")
    out_pixfmts+=("$c5")
    sequence_lens+=("$c6")
    shuffles+=("$c7")
  done < <(tail -n +2 "$TESTCASES")
}

test_body() {

  for i in "${!videos[@]}"; do
      video=${videos[$i]}
      codec=${codecs[$i]}
      gop=${gops[$i]}
      in_pixfmt=${in_pixfmts[$i]}
      out_pixfmt=${out_pixfmts[$i]}
      sequence_len=${sequence_lens[$i]}
      shuffle=${shuffles[$i]}

      echo "video:${video} codec:${codec} gop:${gop} in_pixfmt:${in_pixfmt}" \
            "out_pixfmt:${out_pixfmt} sequence_len:${sequence_len} shuffle:${shuffle}"

      output_file=${video##*/}
      output_file=${output_file%.*}_${gop}_${codec}_${in_pixfmt}.mp4
      echo "Transcoding ${video} with gop size ${gop}...."

      # transcode to correct gop size and codec
      if [ "$codec" == "h264" ]; then
        ffmpeg -y -loglevel fatal -i "$video" -vcodec libx264 -x264-params \
          keyint="${gop}":scenecut=0:no-open-gop=1 -acodec copy -pix_fmt "${in_pixfmt}" \
          "${output_file}";
      elif [ "$codec" == "h265" ]; then
        ffmpeg -y -loglevel fatal -i "$video" -vcodec libx265 -x265-params \
          keyint="${gop}":scenecut=0:no-open-gop=1 -acodec copy -pix_fmt "${in_pixfmt}" \
          "${output_file}";
      else
        echo "Incorrect codec in testcase"
        exit
      fi

      num_frames=$(ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of default=nokey=1:noprint_wrappers=1 "${output_file}")

      # export environment variables for current video
      export DALI_TEST_VIDEO_READER_SEQ_LEN=${sequence_len}
      export DALI_TEST_VIDEO_READER_FILES=${output_file}
      export DALI_TEST_VIDEO_READER_IMAGE_TYPE=${out_pixfmt}
      export DALI_TEST_VIDEO_READER_SHUFFLE=${shuffle}
      export DALI_TEST_VIDEO_BENCH_ITER=$((num_frames / gop - 1))

      # run benchmark app
      DALI_OUTPUT=$("$FULL_PATH" --benchmark_format=csv --benchmark_filter="VideoReaderBench*"); echo "$DALI_OUTPUT"
      DALI_OUTPUT=$(echo "$DALI_OUTPUT" | tail -n 1 | awk -F "," '{printf "%s", $NF}' | grep -Eo '[0-9]+\.[0-9]+')
      echo "${DALI_OUTPUT}," >> "$CSV_LOG_FILE"

  done

}

pushd ../..
source ./qa/test_template.sh
popd

