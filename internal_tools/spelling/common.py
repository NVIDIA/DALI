# Copyright (c) 2023-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import subprocess
import re
from dataclasses import dataclass


CWD = os.getcwd()
CSPELL_DOCKER = "ghcr.io/streetsidesoftware/cspell:latest"
CSPELL_COMMAND = f"docker run -it -v {CWD}:/workdir {CSPELL_DOCKER} \
lint --no-exit-code --config tools/spelling/cspell.json"
FIXME_FILE = "fixme.spelling"
DICT_FILE = "tools/spelling/cspell_dicts/cspell_dict.txt"


def cspell(*args):
    QUIET_FLAGS = ["--no-progress", "--no-color", "--quiet"]
    cmd = CSPELL_COMMAND.split(" ") + QUIET_FLAGS + list(args)
    print("Running cspell:", " ".join(cmd))
    return subprocess.check_output(cmd).decode("utf-8")


@dataclass
class Problem:
    file: str
    line: int
    column: int
    offset: int
    word: str
    fix: str
    ctx: str

    @staticmethod
    def parse(message):
        change_pattern = r"\((?P<word>[^)]+)\)->\((?P<fix>[^)]*)\)"
        loc_pattern = r'"(?P<file>.+)":(?P<offset>\d+):(?P<line>\d+):(?P<col>\d+)'
        pattern = rf"{change_pattern}\s+in {loc_pattern}\s+-- (?P<ctx>.*)"
        m = re.match(pattern, message)
        assert m, f"Unable to parse problem info: {message}"

        return Problem(
            file=m.group("file"),
            offset=int(m.group("offset")),
            line=int(m.group("line")),
            column=int(m.group("col")),
            word=m.group("word"),
            fix=m.group("fix"),
            ctx=m.group("ctx"),
        )

    def export(self):
        loc = f'"{self.file}":{self.offset}:{self.line}:{self.column}'
        return f"({self.word})->({self.fix}) in {loc}  -- {self.ctx}"


def read_lines(path):
    with open(path, "r") as f:
        return [line for line in f.read().split("\n") if line.strip()]
