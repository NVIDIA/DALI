#!/bin/bash

# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FFMPEG_DIR=$1
# See this for the RPATH line: https://stackoverflow.com/questions/71423110/bundling-ffmpeg-using-cmake-and-rpath
LDSOFLAGS=-Wl,-rpath,\''$$$$ORIGIN'\' \
./configure \
  --prefix=${FFMPEG_DIR} \
  --enable-shared \
  --enable-pic \
  --enable-avformat \
  --enable-avcodec \
  --enable-avfilter \
  --enable-avutil \
  --enable-swresample \
  --enable-protocol=file \
  --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb,mpeg4_unpack_bframes \
  --disable-static \
  --disable-programs \
  --disable-doc \
  --disable-avdevice \
  --disable-postproc \
  --disable-w32threads \
  --disable-os2threads \
  --disable-dwt \
  --disable-error-resilience \
  --disable-lsp \
  --disable-faan \
  --disable-pixelutils \
  --disable-autodetect \
  --disable-iconv \
  --disable-encoders \
  --disable-hwaccels \
  --disable-muxers \
  --disable-protocols \
  --disable-indevs \
  --disable-outdevs \
  --disable-devices \
  --disable-filters \
  --disable-bsfs \
  --disable-decoder=ipu
# adds | sed 's/\(.*{\)/DALI_\1/' | to the version file generation command - it prepends "DALI_" to the symbol version
sed -i 's/\$\$(M)sed '\''s\/MAJOR\/\$(lib$(NAME)_VERSION_MAJOR)\/'\'' \$\$< | \$(VERSION_SCRIPT_POSTPROCESS_CMD) > \$\$\@/\$\$(M)sed '\''s\/MAJOR\/\$(lib$(NAME)_VERSION_MAJOR)\/'\'' \$\$< | sed '\''s\/\\(\.*{\\)\/DALI_\\1\/'\'' | \$(VERSION_SCRIPT_POSTPROCESS_CMD) > \$\$\@/' ffbuild/library.mak
make -j install
