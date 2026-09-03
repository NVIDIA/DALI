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

import argparse
import json
import math
import sqlite3
import subprocess
import sys
import tempfile
from collections import defaultdict
from contextlib import closing
from itertools import groupby
from operator import itemgetter
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

# Nsight packs a process ID into the high bits of every global thread ID.
_PID_BITS = 24
_PID_MASK = (1 << _PID_BITS) - 1
_PROCESS_MASK = ~_PID_MASK
_RANGE_TYPES = {"NvtxPushPopRange", "NvtxStartEndRange"}
_INCOMPLETE_RANGE_TYPES = {"NvtxPushRange", "NvtxStartRange"}
_WINDOW_NAME = "profile_window"
_WINDOW_DOMAIN = "data-loading"
_WORKER_DOMAIN = "data-loading-worker"
_REQUIRED_COLUMNS = {
    "StringIds": {"id", "value"},
    "ENUM_NSYS_EVENT_TYPE": {"id", "name"},
    "PROCESSES": {"globalPid", "pid", "name"},
    "NVTX_EVENTS": {
        "start",
        "end",
        "eventType",
        "color",
        "text",
        "globalTid",
        "textId",
        "domainId",
    },
}


class SummaryError(Exception):
    pass


def _sql_list(names: Iterable[str]) -> str:
    """Quote event-type names for an SQL IN clause. They are ASCII literals, never user input."""
    return ", ".join(repr(name) for name in sorted(names))


def _row_pid(row: sqlite3.Row) -> int:
    """The real PID when Nsight recorded the process, else the packed global ID."""
    return row["pid"] if row["pid"] is not None else row["global_pid"]


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="Nsight Systems .nsys-rep")
    parser.add_argument("--output", required=True, type=Path, help="JSON summary path")
    return parser.parse_args()


def _resolve_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    try:
        trace = args.trace.expanduser().resolve(strict=True)
    except OSError as exc:
        raise SummaryError(f"input is unavailable: {exc}") from exc
    output = args.output.expanduser().resolve(strict=False)
    if output == trace:
        raise SummaryError("output must differ from the input trace")
    return trace, output


def _export_sqlite(trace: Path, directory: Path) -> Path:
    output = directory / "trace.sqlite"
    command = [
        "nsys",
        "export",
        "--type=sqlite",
        "--quiet=true",
        "--force-overwrite=true",
        f"--output={output}",
        str(trace),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
    except OSError as exc:
        raise SummaryError(f"failed to run nsys export: {exc}") from exc
    if result.returncode != 0 or not output.is_file():
        detail = (result.stderr or result.stdout).strip()
        raise SummaryError(f"nsys export failed: {detail or f'exit {result.returncode}'}")
    return output


def _schema(connection: sqlite3.Connection) -> dict[str, set[str]]:
    tables = {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }
    required = {"NVTX_EVENTS", "StringIds", "ENUM_NSYS_EVENT_TYPE", "PROCESSES"}
    missing = sorted(required - tables)
    if missing:
        raise SummaryError(f"unsupported Nsight schema; missing tables: {', '.join(missing)}")
    return {
        table: {row[1] for row in connection.execute(f'PRAGMA table_info("{table}")')}
        for table in tables
    }


def _require_columns(schema: dict[str, set[str]], table: str, columns: set[str]) -> None:
    missing = sorted(columns - schema.get(table, set()))
    if missing:
        raise SummaryError(f"unsupported Nsight schema; {table} lacks: {', '.join(missing)}")


def _metadata(connection: sqlite3.Connection, schema: dict[str, set[str]]) -> dict[str, str]:
    if "META_DATA_EXPORT" not in schema or not {"name", "value"} <= schema["META_DATA_EXPORT"]:
        return {}
    return dict(
        connection.execute("SELECT name, value FROM META_DATA_EXPORT WHERE value IS NOT NULL")
    )


def _merge_intervals(intervals: Iterable[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted((start, end) for start, end in intervals if end > start):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def _duration(intervals: Iterable[tuple[int, int]]) -> int:
    return sum(end - start for start, end in _merge_intervals(intervals))


def _intersection_duration(
    left: Iterable[tuple[int, int]], right: Iterable[tuple[int, int]]
) -> int:
    left, right = _merge_intervals(left), _merge_intervals(right)
    total = i = j = 0
    while i < len(left) and j < len(right):
        total += max(0, min(left[i][1], right[j][1]) - max(left[i][0], right[j][0]))
        if left[i][1] < right[j][1]:
            i += 1
        else:
            j += 1
    return total


def _prepare_nvtx(connection: sqlite3.Connection) -> None:
    names = _sql_list(_RANGE_TYPES | _INCOMPLETE_RANGE_TYPES | {"NvtxDomainCreate"})
    connection.executescript(f"""
        CREATE TEMP VIEW normalized_nvtx AS
        WITH raw AS (
          SELECT e.rowid AS event_id, e.start, e.end, e.color,
                 COALESCE(e.text, s.value) AS name, e.domainId,
                 (e.globalTid & {_PROCESS_MASK}) AS global_pid, t.name AS event_type
          FROM NVTX_EVENTS e
          JOIN ENUM_NSYS_EVENT_TYPE t ON t.id = e.eventType
          LEFT JOIN StringIds s ON s.id = e.textId
          WHERE t.name IN ({names})
        ),
        latest_domain AS (
          SELECT global_pid, domainId, MAX(event_id) AS event_id
          FROM raw WHERE event_type = 'NvtxDomainCreate' AND name IS NOT NULL
          GROUP BY global_pid, domainId
        ),
        own_domain AS (
          SELECT r.global_pid, r.domainId, r.name
          FROM raw r JOIN latest_domain d USING (global_pid, domainId, event_id)
        ),
        unambiguous_domain AS (
          SELECT domainId, MIN(name) AS name
          FROM raw WHERE event_type = 'NvtxDomainCreate' AND name IS NOT NULL
          GROUP BY domainId HAVING COUNT(DISTINCT name) = 1
        )
        SELECT r.*, COALESCE(p.pid, (r.global_pid >> {_PID_BITS}) & {_PID_MASK}) AS pid,
               p.name AS process, COALESCE(own.name, any.name) AS domain
        FROM raw r
        LEFT JOIN own_domain own USING (global_pid, domainId)
        LEFT JOIN unambiguous_domain any USING (domainId)
        LEFT JOIN PROCESSES p ON p.globalPid = r.global_pid;
        """)


def _select_window(
    connection: sqlite3.Connection, trace_duration_ns: int
) -> tuple[int, dict[str, Any]]:
    """Find the single measurement window, rejecting absent, unresolved, or duplicate ones."""
    candidates = list(
        connection.execute(
            f"""SELECT * FROM normalized_nvtx
                WHERE event_type IN ({_sql_list(_RANGE_TYPES | _INCOMPLETE_RANGE_TYPES)})
                  AND name = ? AND domain = ?""",
            (_WINDOW_NAME, _WINDOW_DOMAIN),
        )
    )
    incomplete = [
        str(_row_pid(row))
        for row in candidates
        if row["end"] is None or int(row["end"]) > trace_duration_ns
    ]
    if incomplete:
        raise SummaryError(
            f"incomplete {_WINDOW_NAME!r} range for PID(s): {', '.join(sorted(incomplete))}"
        )
    windows = [
        row
        for row in candidates
        if row["event_type"] in _RANGE_TYPES
        and row["end"] is not None
        and int(row["end"]) > int(row["start"])
    ]
    if not windows:
        raise SummaryError(f"no completed {_WINDOW_NAME!r}@{_WINDOW_DOMAIN} range")
    if any(row["pid"] is None for row in windows):
        pids = sorted(str(row["global_pid"]) for row in windows if row["pid"] is None)
        raise SummaryError(f"cannot resolve profile-window process ID(s): {', '.join(pids)}")
    if len(windows) != 1:
        pids = ", ".join(str(row["pid"]) for row in sorted(windows, key=lambda row: row["pid"]))
        raise SummaryError(
            f"expected exactly one {_WINDOW_NAME!r}@{_WINDOW_DOMAIN} range; "
            f"found {len(windows)} for PID(s): {pids}"
        )
    selected = windows[0]
    start, end = int(selected["start"]), int(selected["end"])
    window = {
        "pid": int(selected["pid"]),
        "process": selected["process"],
        "start_ns": start,
        "end_ns": end,
        "duration_ns": end - start,
    }
    return int(selected["global_pid"]), window


def _unnamed_limitations(connection: sqlite3.Connection) -> list[str]:
    """Ranges whose domain ID resolved to no name leave worker coverage indeterminate."""
    unnamed = list(connection.execute(f"""SELECT pid, global_pid FROM normalized_nvtx
                WHERE event_type IN ({_sql_list(_RANGE_TYPES)}) AND end IS NOT NULL AND end > start
                  AND domain IS NULL AND domainId IS NOT NULL AND domainId != 0"""))
    if not unnamed:
        return []
    pids = sorted({str(_row_pid(row)) for row in unnamed})
    return [
        f"{len(unnamed)} completed NVTX range(s) in non-default domains could not be named "
        f"for PID(s): {', '.join(pids)}; domain IDs were absent or ambiguous "
        "across processes."
    ]


def _clip_to_window(
    connection: sqlite3.Connection, global_pid: int, window: dict[str, Any]
) -> None:
    """Expose in-window ranges, clipped to the window, as the `profile_pieces` view."""
    start, end = window["start_ns"], window["end_ns"]
    connection.execute(f"""CREATE TEMP VIEW profile_pieces AS
            SELECT *, MAX(start, {start}) AS clipped_start,
                   MIN(end, {end}) AS clipped_end
            FROM normalized_nvtx
            WHERE event_type IN ({_sql_list(_RANGE_TYPES)}) AND end IS NOT NULL
              AND end > {start} AND start < {end}
              AND domain IN ('{_WINDOW_DOMAIN}', '{_WORKER_DOMAIN}')
              AND (global_pid = {global_pid} OR domain = '{_WORKER_DOMAIN}')""")


def _range_summaries(connection: sqlite3.Connection) -> list[dict[str, Any]]:
    """One entry per (pid, process, domain, name) group, with clipped duration statistics."""
    summaries = []
    rows = connection.execute("""SELECT pid, process, domain, COALESCE(name, '<unnamed>') AS name,
                  color, start, end, clipped_start, clipped_end
           FROM profile_pieces WHERE clipped_end > clipped_start
           ORDER BY domain, pid, name, process""")
    # Grouping needs only contiguity, and ORDER BY covers the same four columns.
    range_key = itemgetter("pid", "process", "domain", "name")
    for (resolved_pid, process, domain, name), entries in groupby(rows, range_key):
        entries = list(entries)
        durations = [int(row["clipped_end"] - row["clipped_start"]) for row in entries]
        partial_count = sum(
            duration != row["end"] - row["start"] for duration, row in zip(durations, entries)
        )
        durations.sort()
        total = sum(durations)
        summaries.append(
            {
                "pid": resolved_pid,
                "process": process,
                "domain": domain,
                "name": name,
                "uncolored_count": sum(row["color"] is None for row in entries),
                "partial_count": partial_count,
                "colors_argb": sorted(
                    {
                        f"#{int(row['color']) & 0xFFFFFFFF:08x}"
                        for row in entries
                        if row["color"] is not None
                    }
                ),
                "count": len(durations),
                "sum_ns": total,
                "mean_ns": total / len(durations),
                # Nearest-rank percentiles: p95 is the maximum below 20 samples.
                "p50_ns": durations[math.ceil(0.50 * len(durations)) - 1],
                "p95_ns": durations[math.ceil(0.95 * len(durations)) - 1],
            }
        )
    return summaries


def _wait_intervals(connection: sqlite3.Connection, global_pid: int) -> list[tuple[int, int]]:
    """Merged main-process loader waits inside the window."""
    return _merge_intervals(
        (int(row[0]), int(row[1]))
        for row in connection.execute(
            """SELECT clipped_start, clipped_end FROM profile_pieces
               WHERE global_pid = ? AND domain = ? AND name = 'batch_wait'
                 AND clipped_end > clipped_start""",
            (global_pid, _WINDOW_DOMAIN),
        )
    )


def _coverage(connection: sqlite3.Connection, summaries: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "pids_by_domain": {
            domain: sorted(
                {
                    item["pid"]
                    for item in summaries
                    if item["domain"] == domain and item["pid"] is not None
                }
            )
            for domain in (_WINDOW_DOMAIN, _WORKER_DOMAIN)
        },
        "unresolved_pid_events": connection.execute(
            "SELECT COUNT(*) FROM profile_pieces WHERE clipped_end > clipped_start AND pid IS NULL"
        ).fetchone()[0],
    }


def _enum_labels(
    connection: sqlite3.Connection,
    schema: dict[str, set[str]],
    table: str,
) -> dict[int, str]:
    if table not in schema or not {"id", "label"} <= schema[table]:
        return {}
    return dict(connection.execute(f'SELECT id, label FROM "{table}"'))


def _enum_label(labels: dict[int, str], value: Any) -> str:
    return labels.get(value, f"Unknown ({value})")


def _load_gpu(
    connection: sqlite3.Connection,
    schema: dict[str, set[str]],
    global_pid: int,
    window: dict[str, Any],
    wait_intervals: list[tuple[int, int]],
) -> tuple[list[dict[str, Any]], list[str]]:
    specifications = {
        "kernel": ("CUPTI_ACTIVITY_KIND_KERNEL", "start end deviceId globalPid"),
        "memcpy": (
            "CUPTI_ACTIVITY_KIND_MEMCPY",
            "start end deviceId globalPid bytes copyKind srcKind dstKind",
        ),
        "memset": ("CUPTI_ACTIVITY_KIND_MEMSET", "start end deviceId globalPid"),
    }
    available: dict[str, tuple[str, list[str]]] = {}
    limitations: list[str] = []
    for kind, (table, column_names) in specifications.items():
        columns = column_names.split()
        if table not in schema or not set(columns) <= schema[table]:
            limitations.append(f"{table} is unavailable; {kind} activity is omitted.")
        else:
            available[kind] = table, columns
    if not available:
        return [], limitations

    window_start, window_end = window["start_ns"], window["end_ns"]
    copy_kinds = _enum_labels(connection, schema, "ENUM_CUDA_MEMCPY_OPER")
    memory_kinds = _enum_labels(connection, schema, "ENUM_CUDA_MEM_KIND")
    intervals: dict[int, dict[str, list[tuple[int, int]]]] = defaultdict(
        lambda: {kind: [] for kind in specifications}
    )
    copy_groups: dict[int, dict[tuple[str, str, str], dict[str, Any]]] = defaultdict(dict)

    for kind, (table, columns) in available.items():
        rows = connection.execute(
            f"""SELECT {", ".join(columns)} FROM "{table}"
                WHERE globalPid = ? AND end > ? AND start < ?""",
            (global_pid, window_start, window_end),
        )
        for row in rows:
            start, end = max(int(row["start"]), window_start), min(int(row["end"]), window_end)
            if end <= start:
                continue
            device_id = int(row["deviceId"])
            intervals[device_id][kind].append((start, end))
            if kind != "memcpy":
                continue
            key = (
                _enum_label(copy_kinds, row["copyKind"]),
                _enum_label(memory_kinds, row["srcKind"]),
                _enum_label(memory_kinds, row["dstKind"]),
            )
            group = copy_groups[device_id].setdefault(
                key,
                {"count": 0, "bytes": 0, "partial_count": 0, "intervals": []},
            )
            group["count"] += 1
            group["bytes"] += int(row["bytes"])
            group["partial_count"] += (end - start) != int(row["end"]) - int(row["start"])
            group["intervals"].append((start, end))

    wait = sum(end - start for start, end in wait_intervals)
    summaries = []
    for device_id in sorted(intervals):
        by_kind = intervals[device_id]
        all_intervals = [interval for values in by_kind.values() for interval in values]
        active = _duration(all_intervals)
        wait_idle = wait - _intersection_duration(wait_intervals, all_intervals)
        copies = []
        for (copy_kind, source, destination), group in sorted(copy_groups[device_id].items()):
            copies.append(
                {
                    "kind": copy_kind,
                    "source_memory": source,
                    "destination_memory": destination,
                    "count": group["count"],
                    "bytes": group["bytes"],
                    "partial_count": group["partial_count"],
                    "sum_ns": sum(end - start for start, end in group["intervals"]),
                    "union_ns": _duration(group["intervals"]),
                }
            )
        duration = window["duration_ns"]
        idle = duration - active
        summaries.append(
            {
                "pid": window["pid"],
                "device_id": device_id,
                "window_duration_ns": duration,
                "active_union_ns": active,
                "idle_ns": idle,
                "active_percent": 100.0 * active / duration,
                "kernel_union_ns": _duration(by_kind["kernel"]),
                "memcpy_union_ns": _duration(by_kind["memcpy"]),
                "memset_union_ns": _duration(by_kind["memset"]),
                "batch_wait_union_ns": wait,
                "batch_wait_gpu_idle_overlap_ns": wait_idle,
                "batch_wait_overlapping_gpu_idle_percent": (
                    100.0 * wait_idle / wait if wait else None
                ),
                "gpu_idle_overlapping_batch_wait_percent": (
                    100.0 * wait_idle / idle if idle else None
                ),
                "copies": copies,
            }
        )
    if not summaries:
        limitations.append(
            "No CUDA kernel, memcpy, or memset events overlapped the selected window(s)."
        )
    return summaries, limitations


def _summarize(connection: sqlite3.Connection, trace: Path) -> dict[str, Any]:
    schema = _schema(connection)
    metadata = _metadata(connection, schema)
    if "ANALYSIS_DETAILS" not in schema or "duration" not in schema["ANALYSIS_DETAILS"]:
        raise SummaryError("unsupported Nsight schema; ANALYSIS_DETAILS.duration is unavailable")
    trace_duration_ns = connection.execute("SELECT MAX(duration) FROM ANALYSIS_DETAILS").fetchone()[
        0
    ]
    if not trace_duration_ns:
        raise SummaryError("Nsight trace duration is unavailable")
    for table, columns in _REQUIRED_COLUMNS.items():
        _require_columns(schema, table, columns)
    _prepare_nvtx(connection)
    global_pid, window = _select_window(connection, int(trace_duration_ns))
    limitations = _unnamed_limitations(connection)
    _clip_to_window(connection, global_pid, window)
    ranges = _range_summaries(connection)
    coverage = _coverage(connection, ranges)
    gpu, gpu_limitations = _load_gpu(
        connection, schema, global_pid, window, _wait_intervals(connection, global_pid)
    )
    return {
        "source": {
            "trace": str(trace),
            "nsys_version": metadata.get("EXPORT_PRODUCT_VERSION"),
            "export_schema_version": metadata.get("EXPORT_SCHEMA_VERSION"),
        },
        "selection": {
            "window_name": _WINDOW_NAME,
            "window_domain": _WINDOW_DOMAIN,
            "worker_domain": _WORKER_DOMAIN,
            "windows": [window],
            "window_union_duration_ns": float(window["duration_ns"]),
        },
        "nvtx_ranges": ranges,
        "gpu": gpu,
        "coverage": coverage,
        "limitations": limitations + gpu_limitations,
    }


def main() -> int:
    args = _arguments()
    try:
        trace, output = _resolve_paths(args)
        with tempfile.TemporaryDirectory(prefix="nsys-summary-") as directory:
            sqlite_path = _export_sqlite(trace, Path(directory))
            uri = f"file:{quote(str(sqlite_path))}?mode=ro"
            with closing(sqlite3.connect(uri, uri=True)) as connection:
                connection.row_factory = sqlite3.Row
                summary = _summarize(connection, trace)
        try:
            with output.open("w", encoding="utf-8") as stream:
                json.dump(summary, stream, indent=2)
                stream.write("\n")
        except OSError as exc:
            raise SummaryError(f"cannot write summary: {exc}") from exc
    except SummaryError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except sqlite3.Error as exc:
        print(f"error: cannot read Nsight SQLite export: {exc}", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "output": str(output),
                "windows": len(summary["selection"]["windows"]),
                "nvtx_groups": len(summary["nvtx_ranges"]),
                "gpu_groups": len(summary["gpu"]),
            },
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
