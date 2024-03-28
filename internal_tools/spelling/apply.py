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

from common import Problem, FIXME_FILE, DICT_FILE, read_lines
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f"Apply corrections of spelling mistakes from {FIXME_FILE} "
        "and update the dictionary."
    )
    parser.parse_args()

    skipped = set()

    problems = [Problem.parse(line) for line in read_lines(FIXME_FILE)]

    problems_in_file = dict()
    for p in problems:
        if not p.fix:
            skipped.add(p.word.lower())
        else:
            problems_in_file[p.file] = problems_in_file.get(p.file, []) + [p]

    fixed = 0
    for file, problems in problems_in_file.items():
        current_offset = 0
        with open(file, "r") as f:
            text = f.read()
        new_text = ""
        skip_file = False
        for p in sorted(problems, key=lambda p: p.offset):
            new_text += text[current_offset : p.offset]
            orig = text[p.offset : p.offset + len(p.word)]
            if orig != p.word:
                print(
                    f"In {p.file}:{p.line}, when applying ({p.word})->({p.fix}): "
                    f'expected "{p.word}" in the file, but there\'s "{orig}". Skipping entire file!'
                )
                skip_file = True
                break
            new_text += p.fix
            current_offset = p.offset + len(p.word)
            fixed += 1
        new_text += text[current_offset:]
        if not skip_file:
            with open(file, "w") as f:
                f.write(new_text)

    print(f"Applied {fixed} fixes.")

    old_dict = {w.lower() for w in read_lines(DICT_FILE)}
    new_dict = old_dict | skipped
    with open(DICT_FILE, "w") as f:
        f.write("\n".join(w for w in sorted(new_dict)) + "\n")
    print(f"{len(new_dict) - len(old_dict)} words were added to the dictionary.")
