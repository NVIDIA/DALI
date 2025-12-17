# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import json
from pathlib import Path


def find_notebook_pairs(examples_dir: Path) -> dict:
    """
    Scan the examples directory for notebook pairs.

    A pair is identified when:
    - `foo.ipynb` exists AND `foo_dynamic.ipynb` exists

    Returns a dict mapping of each HTML page path to its variant path.
    """
    variants = {}

    all_notebooks = list(examples_dir.rglob("*.ipynb"))

    notebook_stems: dict[str, Path] = {}
    for nb in all_notebooks:
        rel_path = nb.relative_to(examples_dir)
        notebook_stems[rel_path.stem] = rel_path

    for stem, rel_path in notebook_stems.items():
        if stem.endswith("_dynamic"):
            pipeline_stem = stem.removeprefix("_dynamic")
            if pipeline_stem in notebook_stems:
                dynamic_html = f"examples/{stem}.html"
                pipeline_html = f"examples/{pipeline_stem}.html"

                variants[dynamic_html] = pipeline_html
                variants[pipeline_html] = dynamic_html

    return variants


def generate(output_path: Path):
    """Generate the mode variants manifest JSON file."""
    examples_dir = Path(__file__).parent / "examples"

    variants = find_notebook_pairs(examples_dir)
    manifest = {"variants": variants}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Generated mode variants manifest with {len(variants)} entries: {output_path}"
    )
    return manifest


if __name__ == "__main__":
    # For testing: generate to stdout
    import sys

    if len(sys.argv) > 1:
        output = Path(sys.argv[1])
    else:
        output = Path("_static/mode_variants.json")

    generate(output)
