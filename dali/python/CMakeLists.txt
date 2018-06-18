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
file(GLOB tmp *.cc)
set(dali_python_lib "backend_impl")
pybind11_add_module(${dali_python_lib} MODULE ${tmp})
add_dependencies(${dali_python_lib} ${dali_lib})
if (BUILD_TENSORFLOW)
  add_dependencies(${dali_python_lib} ${customop_lib})
endif()
target_link_libraries(${dali_python_lib} PRIVATE ${dali_lib} ${DALI_LIBS})
set_target_properties(${dali_python_lib} PROPERTIES
  LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/dali/python/nvidia/dali)

# Add install rule
install(DIRECTORY "${PROJECT_SOURCE_DIR}/dali/python/nvidia" DESTINATION "${PROJECT_BINARY_DIR}/dali/python")
install(FILES "${PROJECT_SOURCE_DIR}/dali/python/setup.py" DESTINATION "${PROJECT_BINARY_DIR}/dali/python")
install(FILES "${PROJECT_SOURCE_DIR}/tools/rec2idx.py" DESTINATION "${PROJECT_BINARY_DIR}/dali/python")
install(FILES "${PROJECT_SOURCE_DIR}/tools/tfrecord2idx" DESTINATION "${PROJECT_BINARY_DIR}/dali/python")

# Get the COPYRIGHT, LICENSE, and Acknowledgements
install(FILES "${PROJECT_SOURCE_DIR}/Acknowledgements.txt" DESTINATION "${PROJECT_BINARY_DIR}/dali/python/nvidia/dali")
install(FILES "${PROJECT_SOURCE_DIR}/COPYRIGHT" DESTINATION "${PROJECT_BINARY_DIR}/dali/python/nvidia/dali")
install(FILES "${PROJECT_SOURCE_DIR}/LICENSE" DESTINATION "${PROJECT_BINARY_DIR}/dali/python/nvidia/dali")
