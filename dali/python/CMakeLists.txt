# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

# Get all the srcs
collect_headers(DALI_INST_HDRS PARENT_SCOPE)
collect_sources(DALI_PYTHON_BACKEND_SRCS)

build_per_python_lib(dali_python
                     OUTPUT_NAME backend_impl
                     OUTPUT_DIR ${DALI_LIBRARY_OUTPUT_DIR}
                     PUBLIC_LIBS dali dali_operators dali_kernels dali_core ${CUDART_LIB}
                     PRIV_LIBS ${CUDA_LIBRARIES} dynlink_cuda
                     EXCLUDE_LIBS ${exclude_libs}
                     SRC ${DALI_PYTHON_BACKEND_SRCS})

if (DALI_BUILD_FLAVOR)
  set(DALI_FLAVOR "${DALI_BUILD_FLAVOR} ")
  set(DALI_FLAVOR_MINUS "-${DALI_BUILD_FLAVOR}")
  set(DALI_VERSION "${DALI_VERSION}.${TIMESTAMP}")
endif()

# Add the COPYRIGHT, LICENSE, and Acknowledgements
copy_post_build(dali_python "${PROJECT_SOURCE_DIR}/dali/python/nvidia" "${PROJECT_BINARY_DIR}/dali/python")
configure_file("${PROJECT_SOURCE_DIR}/dali/python/__init__.py.in" "${PROJECT_BINARY_DIR}/dali/python/nvidia/dali/__init__.py")
configure_file("${PROJECT_SOURCE_DIR}/dali/python/setup.py.in" "${PROJECT_BINARY_DIR}/stage/setup.py")
copy_post_build(dali_python "${PROJECT_BINARY_DIR}/stage/setup.py" "${PROJECT_BINARY_DIR}/dali/python")
copy_post_build(dali_python "${PROJECT_SOURCE_DIR}/dali/python/MANIFEST.in" "${PROJECT_BINARY_DIR}/dali/python")
copy_post_build(dali_python "${PROJECT_SOURCE_DIR}/tools/rec2idx.py" "${PROJECT_BINARY_DIR}/dali/python")
copy_post_build(dali_python "${PROJECT_SOURCE_DIR}/tools/tfrecord2idx" "${PROJECT_BINARY_DIR}/dali/python")
copy_post_build(dali_python "${PROJECT_SOURCE_DIR}/Acknowledgements.txt" "${PROJECT_BINARY_DIR}/dali/python/nvidia/dali")
copy_post_build(dali_python "${PROJECT_SOURCE_DIR}/COPYRIGHT" "${PROJECT_BINARY_DIR}/dali/python/nvidia/dali")
copy_post_build(dali_python "${PROJECT_SOURCE_DIR}/LICENSE" "${PROJECT_BINARY_DIR}/dali/python/nvidia/dali")
copy_post_build(dali_python "${PROJECT_SOURCE_DIR}/tools/find_dali.cmake" "${PROJECT_BINARY_DIR}/dali/cmake")
