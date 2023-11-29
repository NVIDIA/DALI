#!/usr/bin/env python3
# Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
from pathlib import Path

from nvidia.dali.ops import _signatures

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Produces .pyi stub files for nvidia.dali.fn and nvidia.dali.ops modules"
    )
    parser.add_argument(
        "--wheel_path",
        type=str,
        help=(
            "Path to the `nvidia/dali/` directory inside the wheel, before packaging,"
            ' for example: "/opt/dali/build/dali/python/nvidia/dali".'
        ),
    )
    args = parser.parse_args()

    print(f"Generating signatures for {args.wheel_path=}")

    _signatures.gen_all_signatures(Path(args.wheel_path), "fn")
    _signatures.gen_all_signatures(Path(args.wheel_path), "ops")
