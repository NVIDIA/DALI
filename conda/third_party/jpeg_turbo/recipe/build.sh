#!/bin/bash
#
# (C) Copyright IBM Corp. 2019. All Rights Reserved.
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


# Create build directory for cmake and enter it
mkdir $SRC_DIR/build
cd $SRC_DIR

# Configure the build to install in unique location
autoreconf -fiv
cd build
sh $SRC_DIR/configure --prefix $PREFIX \
                      --bindir $PREFIX/bin \
                      --libdir $PREFIX/lib \
                      --includedir $PREFIX/include \
                      --mandir $PREFIX/man

# Build
make -j"$(nproc --all)"
make install
