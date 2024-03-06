# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This project creates the project to pack as a source distribution
file(COPY "pkg_src" DESTINATION ".") # copy to build folder
file(COPY "src" DESTINATION "pkg_src") # copy to build folder
file(COPY "${DALI_ROOT}/Acknowledgements.txt" DESTINATION "pkg_src")
file(COPY "${DALI_ROOT}/COPYRIGHT" DESTINATION "pkg_src")
file(COPY "${DALI_ROOT}/LICENSE" DESTINATION "pkg_src")
configure_file("setup.py.in" "pkg_src/setup.py")
configure_file("src/nvidia/dali_video/__init__.py.in" "pkg_src/src/nvidia/dali_video/__init__.py")

add_custom_target(dali_video ALL)
add_custom_command(
    TARGET dali_video
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pkg_src
    COMMAND ${PYTHON_EXECUTABLE} setup.py sdist)

install(DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/pkg_src/dist
        DESTINATION . FILES_MATCHING PATTERN "*.tar.gz")
