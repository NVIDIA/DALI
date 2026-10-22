#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Checks the results of internal_tools/video_reader_bench.py (--json output).

Checks:
  * no configuration failed
  * hardware-independent relative checks, with a tolerance:
      - lanes, cpu: throughput at the highest num_threads is not lower than at the
        lowest one (more threads must not make the reader slower)
      - reader_depth: a deeper reader queue is not slower than depth 1
  * optional absolute minimums (frames/s) for named metrics, passed with --min:
      lanes          lanes, batch 16, highest num_threads
      single_stream  single_stream, h264_long
      cpu            cpu, batch 8, highest num_threads
      engines        engines, highest number of readers
    Metrics without a minimum are printed for reference only.

Results flagged as contaminated (other processes used the GPU) only produce warnings.
Exits with 1 if any check failed.
"""

import argparse
import json
import sys
import warnings


def by(results, scenario, **match):
    """DALI results only; results of other libraries (--backends) are for reference."""
    return [
        r
        for r in results
        if r.get("scenario") == scenario
        and r.get("backend", "dali") == "dali"
        and all(r.get(k) == v for k, v in match.items())
    ]


def groups(results, scenario, keys):
    out = {}
    for r in by(results, scenario):
        out.setdefault(tuple(r.get(k) for k in keys), []).append(r)
    return out


def named_metrics(results):
    """The metrics that can have absolute minimums."""

    def at_max(rows, key):
        rows = [r for r in rows if r.get("frames_per_s") is not None]
        return max(rows, key=lambda r: r[key])["frames_per_s"] if rows else None

    return {
        "lanes": at_max(by(results, "lanes", batch=16), "threads"),
        "single_stream": next(
            (r.get("frames_per_s") for r in by(results, "single_stream", files="h264_long")), None
        ),
        "cpu": at_max(by(results, "cpu", batch=8), "threads"),
        "engines": at_max(by(results, "engines"), "readers"),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("results", help="--json output of video_reader_bench.py")
    parser.add_argument("--tolerance", type=float, default=10.0, help="percent, relative checks")
    parser.add_argument(
        "--min", action="append", default=[], metavar="NAME=FPS", help="absolute minimum"
    )
    args = parser.parse_args()

    with open(args.results) as f:
        results = json.load(f)["results"]
    metrics = named_metrics(results)
    minimums = {}
    for item in args.min:
        name, _, value = item.partition("=")
        if name not in metrics:
            parser.error(f"--min got unknown metric {name!r}; expected one of {list(metrics)}.")
        if value:  # an empty value means: no minimum, report only
            try:
                minimums[name] = float(value)
            except ValueError:
                parser.error(f"--min {name} expects a number of frames/s; got {value!r}.")
    tol = args.tolerance / 100
    failed = []

    def check(name, ok, detail):
        print(f"CHECK {name}: {'OK' if ok else 'FAIL'} ({detail})")
        if not ok:
            failed.append(name)

    # a configuration fails if all its repetitions failed (`error`) or only some (`errors`)
    errors = [r for r in results if r.get("error") or r.get("errors")]
    check("no failed configurations", not errors, f"{len(errors)} of {len(results)} failed")
    for r in errors:
        for e in [r["error"]] if r.get("error") else r["errors"]:
            print(f"  {r.get('scenario')} {r.get('files')} {r.get('layout')}: {e}")

    contaminated = [r for r in results if r.get("contaminated")]
    if contaminated:
        warnings.warn(
            f"{len(contaminated)} of {len(results)} results were measured while other "
            f"processes used the GPU; their numbers may be skewed."
        )

    def relative(scenario, keys, param):
        for key, rows in sorted(groups(results, scenario, keys).items(), key=str):
            rows = sorted((r for r in rows if r.get("frames_per_s")), key=lambda r: r[param])
            if len(rows) < 2:
                continue
            base = rows[0]
            for r in rows[1:]:
                ok = r["frames_per_s"] >= (1 - tol) * base["frames_per_s"]
                label = "/".join(str(k) for k in key)
                check(
                    f"{scenario} {label} {param}={r[param]} vs {base[param]}",
                    ok,
                    f"{r['frames_per_s']:.1f} vs {base['frames_per_s']:.1f} frames/s, "
                    f"tolerance {args.tolerance:g}%",
                )

    relative("lanes", ("files", "layout", "batch"), "threads")
    relative("cpu", ("files", "layout", "batch"), "threads")
    relative("reader_depth", ("files", "layout"), "reader_depth")

    for name, value in metrics.items():
        if name in minimums:
            check(
                f"{name} minimum",
                value is not None and value >= minimums[name],
                f"{value} vs min {minimums[name]:g} frames/s",
            )
        else:
            print(f"REFERENCE {name}: {value} frames/s (no minimum set)")

    print("all checks passed" if not failed else f"{len(failed)} check(s) failed")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
