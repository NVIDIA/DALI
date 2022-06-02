# Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

add_custom_target(lint-cpp
        COMMAND
          python ${PROJECT_SOURCE_DIR}/tools/lint.py ${PROJECT_SOURCE_DIR} --nproc=5
        COMMENT
          "Performing C++ linter check"
)

set(PYTHON_LINT_PATHS
        ${PROJECT_SOURCE_DIR}/dali
        ${PROJECT_SOURCE_DIR}/tools
)

add_custom_target(lint-python
        COMMAND
          flake8 --config=${PROJECT_SOURCE_DIR}/.flake8 ${PYTHON_LINT_PATHS}
        COMMENT
          "Performing Python linter check"
)

add_custom_target(lint)
add_dependencies(lint lint-cpp)  #TODO(mszolucha): add lint-python when linter errors are resolved
