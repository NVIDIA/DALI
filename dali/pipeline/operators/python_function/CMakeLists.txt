# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

set(python_function_lib "python_function_plugin")
add_library(${python_function_lib} SHARED python_function.cc)
target_link_libraries(${python_function_lib} ${dali_lib})
target_include_directories(${python_function_lib}
        PRIVATE ${PYBIND11_INCLUDE_DIR}  # from project CMakeLists.txt
        PRIVATE ${pybind11_INCLUDE_DIR}  # from pybind11Config
        PRIVATE ${PYTHON_INCLUDE_DIRS})

# Install python operator plugin to the wheel directory
set_target_properties(${python_function_lib} PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/${DALI_WHEEL_DIR}/")
