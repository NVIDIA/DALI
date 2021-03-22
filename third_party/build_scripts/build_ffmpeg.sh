#!/bin/bash -xe

if [ ${WITH_FFMPEG} -gt 0 ]; then
    # FFmpeg  https://developer.download.nvidia.com/compute/redist/nvidia-dali/ffmpeg-4.3.1.tar.bz2
    pushd third_party/FFmpeg
    ./configure \
        --prefix=${INSTALL_PREFIX} \
        --disable-static \
        --disable-programs \
        --disable-doc \
        --disable-avdevice \
        --disable-swresample \
        --disable-swscale \
        --disable-postproc \
        --disable-w32threads \
        --disable-os2threads \
        --disable-dct \
        --disable-dwt \
        --disable-error-resilience \
        --disable-lsp \
        --disable-lzo \
        --disable-mdct \
        --disable-rdft \
        --disable-fft \
        --disable-faan \
        --disable-pixelutils \
        --disable-autodetect \
        --disable-iconv \
        --enable-shared \
        --enable-avformat \
        --enable-avcodec \
        --enable-avfilter \
        --disable-encoders \
        --disable-hwaccels \
        --disable-muxers \
        --disable-protocols \
        --enable-protocol=file \
        --disable-indevs \
        --disable-outdevs  \
        --disable-devices \
        --disable-filters \
        --disable-bsfs \
        --enable-bsf=h264_mp4toannexb,hevc_mp4toannexb,mpeg4_unpack_bframes
    make -j"$(grep ^processor /proc/cpuinfo | wc -l)"
    make install
    popd
fi
