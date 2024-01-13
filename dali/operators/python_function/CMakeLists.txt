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

collect_headers(DALI_INST_HDRS PARENT_SCOPE)
collect_sources(PYTHON_FUNCTION_SRCS PARENT_SCOPE)
collect_test_sources(DALI_OPERATOR_TEST_SRCS PARENT_SCOPE)


if (PREBUILD_DALI_LIBS)
  # find prebuild DALI libs
  find_library(PREBUILD_DALI_LIB NAMES dali)
  build_per_python_lib(${dali_python_function_lib}
                      OUTPUT_NAME ${dali_python_function_lib}
                      OUTPUT_DIR ${DALI_LIBRARY_OUTPUT_DIR}
                      PUBLIC_LIBS ${PREBUILD_DALI_LIB}
                      PRIV_LIBS ${DALI_LIBS}
                      EXCLUDE_LIBS ${exclude_libs}
                      SRC ${PYTHON_FUNCTION_SRCS})
else(PREBUILD_DALI_LIBS)
  build_per_python_lib(${dali_python_function_lib}
                      OUTPUT_NAME ${dali_python_function_lib}
                      OUTPUT_DIR ${DALI_LIBRARY_OUTPUT_DIR}
                      PUBLIC_LIBS dali
                      PRIV_LIBS ${DALI_LIBS}
                      EXCLUDE_LIBS ${exclude_libs}
                      SRC ${PYTHON_FUNCTION_SRCS})
endif(PREBUILD_DALI_LIBS)
