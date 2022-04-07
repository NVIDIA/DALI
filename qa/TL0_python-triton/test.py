# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import sys
import subprocess

serialized_filename = "some_custom_name"


def test_correct_usage():
    cmd = ["python", "decorated_function.py", serialized_filename, "autoserialize.me"]
    out = subprocess.run(cmd, capture_output=True, cwd=None, check=True, text=True)
    if out.stdout != f"Serialized to {serialized_filename}\n":
        print(out)
        sys.exit(1)


def test_incorrect_usage():
    test_cases = [
        ["python", "decorated_function.py"],
        ["python", "decorated_function.py", "autoserialize.me"],
        ["python", "decorated_function.py", serialized_filename],
        ["python", "decorated_function.py", "autoserialize.me", serialized_filename],
        ["python", "decorated_function.py", serialized_filename, "some_string"],
    ]
    for cmd in test_cases:
        out = subprocess.run(cmd, capture_output=True, cwd=None, check=True, text=True)
        if len(out.stdout) != 0:
            print(out)
            sys.exit(1)


def test_running_triton_py():
    import nvidia.dali.triton as triton
    cmd = ["python", triton.__file__]
    out = subprocess.run(cmd, capture_output=True, cwd=None, check=True, text=True)
    if len(out.stdout) != 0:
        print(out)
        sys.exit(1)


def test_importing_decorated_function():
    cmd = ["python", "imports_decorated_function.py"]
    out = subprocess.run(cmd, capture_output=True, cwd=None, check=True, text=True)
    if len(out.stdout) != 0:
        print(out)
        sys.exit(1)


def test_double_decorated_functions():
    cmd = ["python", "double_decorated_function.py", serialized_filename, "autoserialize.me"]
    try:
        subprocess.run(cmd, capture_output=True, cwd=None, check=True, text=True)
    except subprocess.CalledProcessError:
        return
    sys.exit(1)
