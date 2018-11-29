#!/bin/bash -e

pushd ../..

cd docs/examples/video

# Install FFmpeg
apt-get -y install yasm
FFMPEG_VERSION=3.4.2
wget -q http://ffmpeg.org/releases/ffmpeg-${FFMPEG_VERSION}.tar.bz2 
tar xf ffmpeg-$FFMPEG_VERSION.tar.bz2
rm ffmpeg-$FFMPEG_VERSION.tar.bz2
cd ffmpeg-$FFMPEG_VERSION
./configure \
  --prefix=/usr/local \
  --disable-static \
  --disable-all \
  --disable-autodetect \
  --disable-iconv \
  --enable-shared \
  --enable-avformat \
  --enable-avcodec \
  --enable-avfilter \
  --enable-protocol=file \
  --enable-demuxer=mov,matroska \
  --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb
make -j"$(grep ^processor /proc/cpuinfo | wc -l)" && make install
rm -rf ffmpeg-$FFMPEG_VERSION

# Download video sample
wget -q -O prepared.mp4 https://download.blender.org/durian/trailer/sintel_trailer-720p.mp4

test_body() {
    # test code
    ls
    python video_example.py
}

source ../../../qa/test_template.sh

popd
