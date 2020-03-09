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

mkdir -p build
cd build

# Force -std=c++14 in CXXFLAGS
export CXXFLAGS=${CXXFLAGS/-std=c++??/-std=c++14}

cmake -DCMAKE_BUILD_TYPE=RELEASE \
    -DCMAKE_PREFIX_PATH=${PREFIX} \
    -DCMAKE_INSTALL_PREFIX=${PREFIX} \
    -DCMAKE_INSTALL_LIBDIR="lib" \
    -DBUILD_SHARED_LIBS=OFF \
    -DWITH_CUDA=OFF -DWITH_1394=OFF -DWITH_IPP=OFF -DWITH_OPENCL=OFF -DWITH_GTK=OFF \
    -DBUILD_JPEG=OFF -DWITH_JPEG=ON \
    -DBUILD_TIFF=OFF -DWITH_TIFF=ON \
    -DBUILD_DOCS=OFF -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_PNG=ON \
    -DBUILD_opencv_cudalegacy=OFF -DBUILD_opencv_stitching=OFF \
    -DWITH_TBB=OFF -DWITH_OPENMP=OFF -DWITH_PTHREADS_PF=OFF -DWITH_CSTRIPES=OFF ..

make -j install
