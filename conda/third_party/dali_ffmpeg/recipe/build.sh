# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

# unset the SUBDIR variable since it changes the behavior of make here
unset SUBDIR
./configure \
    --prefix=${PREFIX} \
    --cc=${CC} \
    --disable-static \
    --disable-all \
    --disable-autodetect \
    --disable-iconv \
    --enable-shared \
    --enable-avformat \
    --enable-avcodec \
    --enable-avfilter \
    --enable-protocol=file \
    --enable-demuxer=mov,matroska,avi  \
    --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb,mpeg4_unpack_bframes
make -j"$(nproc --all)"
make install
