# Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
          python ${PROJECT_SOURCE_DIR}/internal_tools/lint.py ${PROJECT_SOURCE_DIR} --nproc=5
        COMMENT
          "Performing C++ linter check"
)

set(PYTHON_SECURITY_LINT_PATHS
        ${PROJECT_SOURCE_DIR}/dali/python
        ${PROJECT_SOURCE_DIR}/tools
        ${PROJECT_SOURCE_DIR}/dali_tf_plugin
)

set (PYTHON_LINT_DOCS_PATHS
        ${PROJECT_SOURCE_DIR}/docs
)
set(PYTHON_LINT_PATHS
        ${PYTHON_SECURITY_LINT_PATHS}
        ${PROJECT_SOURCE_DIR}/dali
        ${PROJECT_SOURCE_DIR}/qa
        ${PROJECT_SOURCE_DIR}/internal_tools
)

set(AUTOGRAPH_LINT_PATHS
        ${PROJECT_SOURCE_DIR}/dali/python/nvidia/dali/_autograph
        ${PROJECT_SOURCE_DIR}/dali/test/python/autograph/
)

add_custom_target(lint-python-black
        # keep black invocation  separated so each invocation will pick appropriate configuration
        # file from the top dir used for it
        COMMAND
          black --check ${PYTHON_LINT_PATHS} ${AUTOGRAPH_LINT_PATHS}
        COMMAND
          black --check ${PYTHON_LINT_DOCS_PATHS}
        COMMENT
          "Performing black Python formatting check"
)

add_custom_target(lint-python-bandit
        COMMAND
          bandit --config ${PROJECT_SOURCE_DIR}/bandit.yml -r ${PYTHON_SECURITY_LINT_PATHS}
        COMMENT
          "Performing Bandit Python security check"
)


add_custom_target(lint-python-flake
        COMMAND
          flake8 --config=${PROJECT_SOURCE_DIR}/.flake8 ${PYTHON_LINT_PATHS} ${PYTHON_LINT_DOCS_PATHS} ${PYTHON_LINT_DOCS_PATHS}
        COMMAND
          flake8 --config=${PROJECT_SOURCE_DIR}/.flake8.ag ${AUTOGRAPH_LINT_PATHS}
        COMMENT
          "Performing Python linter check"
)

add_custom_target(lint-python)
add_dependencies(lint-python lint-python-black lint-python-flake lint-python-bandit)

add_custom_target(lint)
add_dependencies(lint lint-cpp lint-python)
