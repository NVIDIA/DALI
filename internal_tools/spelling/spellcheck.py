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

from common import cspell, Problem, FIXME_FILE
import json
import re

import argparse


def from_cspell_issue(issue):
    return Problem(
        file=re.match("file:///workdir/(.*)", issue["uri"]).group(1),
        word=issue["text"],
        line=issue["row"],
        column=issue["col"],
        ctx=issue["context"]["text"],
        offset=issue["offset"],
        fix="",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Check the code for spelling mistakes and generate {FIXME_FILE}."
    )
    parser.parse_args()

    cspell_out = json.loads(cspell("--show-context", "--reporter", "@cspell/cspell-json-reporter"))

    problems = [from_cspell_issue(issue) for issue in cspell_out["issues"]]
    print(f"Found {len(problems)} problems.")

    counts = {}
    for problem in problems:
        lower = problem.word.lower()
        counts[lower] = counts.get(lower, 0) + 1
    problems = list(sorted(problems, key=lambda p: (counts[p.word.lower()], p.word.lower())))

    print(f"Writing the problems to file: {FIXME_FILE}")
    with open(FIXME_FILE, "w") as f:
        f.write("\n".join(p.export() for p in problems) + "\n")

    print(f"Please now review {FIXME_FILE} and then run apply.py to apply the fixes")
