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
import importlib
import importlib.metadata
import json
import os
import platform
import resource
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _error(exc: BaseException) -> str:
    return f"{type(exc).__name__}: {exc}"


def _limit(value: int) -> int | str:
    return "unlimited" if value == resource.RLIM_INFINITY else value


def _space(path: Path) -> dict[str, int] | None:
    try:
        usage = shutil.disk_usage(path)
    except OSError:
        return None
    return {"total_bytes": usage.total, "available_bytes": usage.free}


def _unescape_mount(value: str) -> str:
    for escaped, plain in (("\\040", " "), ("\\011", "\t"), ("\\012", "\n"), ("\\134", "\\")):
        value = value.replace(escaped, plain)
    return value


def _mount_info(path: Path) -> dict[str, str] | None:
    try:
        resolved = path.resolve(strict=False)
        best: tuple[int, dict[str, str]] | None = None
        for line in Path("/proc/self/mountinfo").read_text().splitlines():
            before, after = line.split(" - ", 1)
            left, right = before.split(), after.split()
            mount_point = Path(_unescape_mount(left[4]))
            try:
                resolved.relative_to(mount_point)
            except ValueError:
                continue
            info = {
                "mount_point": str(mount_point),
                "filesystem": right[0],
                "source": _unescape_mount(right[1]),
            }
            candidate = len(str(mount_point)), info
            if best is None or candidate[0] > best[0]:
                best = candidate
        return best[1] if best else None
    except (OSError, ValueError, IndexError):
        return None


def _path_info(path: Path) -> dict[str, Any]:
    exists = path.exists()
    probe = path if exists else path.parent
    is_directory = path.is_dir()
    # Directories need +x to be usable, and a missing path is judged by its parent directory.
    execute = 0 if exists and not is_directory else os.X_OK
    return {
        "path": str(path.absolute()),
        "exists": exists,
        "is_directory": is_directory,
        "readable": exists and os.access(path, os.R_OK | execute),
        "writable": os.access(probe, os.W_OK | execute),
        "space": _space(probe),
        "mount": _mount_info(probe),
    }


def _command_probe(command: str, args: list[str]) -> dict[str, Any]:
    executable = shutil.which(command)
    if not executable:
        return {"available": False, "executable": None}
    try:
        proc = subprocess.run(
            [executable, *args], capture_output=True, text=True, timeout=15, check=False
        )
        text = (proc.stdout or proc.stderr).strip()
        return {
            "available": proc.returncode == 0,
            "executable": executable,
            "returncode": proc.returncode,
            "version": text,
        }
    except (OSError, subprocess.SubprocessError) as exc:
        return {"available": False, "executable": executable, "error": _error(exc)}


def _module_probe(
    module: str,
    distribution_names: tuple[str, ...] = (),
    required_attribute: str | None = None,
) -> dict[str, Any]:
    try:
        loaded = importlib.import_module(module)
        if required_attribute is not None and not hasattr(loaded, required_attribute):
            return {
                "available": False,
                "module": module,
                "error": f"Missing required attribute: {required_attribute}",
            }
        version = getattr(loaded, "__version__", None)
        if version is None:
            for distribution in distribution_names:
                try:
                    version = importlib.metadata.version(distribution)
                    break
                except importlib.metadata.PackageNotFoundError:
                    pass
        return {
            "available": True,
            "module": module,
            "required_attribute": required_attribute,
            "version": version,
            "path": getattr(loaded, "__file__", None),
        }
    except Exception as exc:
        return {"available": False, "module": module, "error": _error(exc)}


def _torch_probe() -> dict[str, Any]:
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        devices = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "memory_mib": torch.cuda.get_device_properties(index).total_memory >> 20,
            }
            for index in (range(torch.cuda.device_count()) if cuda_available else ())
        ]
        return {
            "available": True,
            "version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "cuda_available": cuda_available,
            "visible_devices": devices,
        }
    except Exception as exc:
        return {
            "available": False,
            "error": _error(exc),
            "cuda_available": False,
            "visible_devices": [],
        }


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    data = parser.add_mutually_exclusive_group(required=True)
    data.add_argument("--data-path", type=Path, help="local dataset path")
    data.add_argument("--data-source", help="remote or custom dataset source")
    parser.add_argument("--source-dir", required=True, type=Path, help="production Git checkout")
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--expected-visible-gpus", type=int, help="minimum visible GPU count")
    args = parser.parse_args()
    if args.data_source is not None and not args.data_source.strip():
        parser.error("--data-source cannot be empty")
    if args.expected_visible_gpus is not None and args.expected_visible_gpus < 1:
        parser.error("--expected-visible-gpus must be at least 1")
    return args


def main() -> int:
    args = _arguments()
    blockers: list[dict[str, str]] = []
    warnings: list[dict[str, str]] = []

    def block(code: str, message: str) -> None:
        blockers.append({"code": code, "message": message})

    try:
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        print(
            f"preflight blocked: cannot create artifact directory: {_error(exc)}", file=sys.stderr
        )
        return 2

    data = (
        _path_info(args.data_path)
        if args.data_path is not None
        else {"kind": "remote_or_custom", "description": args.data_source}
    )
    source = _path_info(args.source_dir)
    artifact = _path_info(args.artifact_dir)
    shared_memory = _path_info(Path("/dev/shm"))
    torch_info = _torch_probe()
    git = _command_probe("git", ["--version"])

    if args.data_path is not None and (not data["exists"] or not data["readable"]):
        block("data_unavailable", "Data path is missing or unreadable.")
    source_ready = source["exists"] and source["is_directory"] and source["readable"]
    if not source_ready:
        block("source_unavailable", "Source directory is missing, unreadable, or not a directory.")
    if not git["available"]:
        block("git_unavailable", "Git is unavailable or its probe failed.")
    if git["available"] and source_ready:
        repo = _command_probe("git", ["-C", str(args.source_dir), "rev-parse", "--show-toplevel"])
        if repo["available"]:
            source["git_root"] = repo["version"]
        else:
            block("source_not_git", "Source directory is not inside a Git worktree.")
    if not torch_info["available"]:
        block("torch_unavailable", "PyTorch cannot be imported by the production Python.")
    elif not torch_info["cuda_available"]:
        block("cuda_unavailable", "CUDA is not visible to the production Python.")
    if (
        torch_info["cuda_available"]
        and args.expected_visible_gpus is not None
        and len(torch_info["visible_devices"]) < args.expected_visible_gpus
    ):
        block("insufficient_visible_gpus", "Visible GPU count is smaller than expected.")

    dali = _module_probe(
        "nvidia.dali.plugin.pytorch.loader_evaluator",
        ("nvidia-dali-cuda130", "nvidia-dali-cuda120", "nvidia-dali"),
        "LoaderEvaluator",
    )
    nvtx = _module_probe("nvtx", ("nvtx",))
    nsys = _command_probe("nsys", ["--version"])
    for code, label, probe in (
        ("dali_unavailable", "DALI replay", dali),
        ("nvtx_unavailable", "NVTX", nvtx),
        ("nsys_unavailable", "Nsight Systems", nsys),
    ):
        if not probe["available"]:
            warnings.append(
                {"code": code, "message": f"{label} is unavailable or its probe failed."}
            )
    nofile = resource.getrlimit(resource.RLIMIT_NOFILE)
    memlock = resource.getrlimit(resource.RLIMIT_MEMLOCK)
    result = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": "blocked" if blockers else "ready",
        "hard_blockers": blockers,
        "warnings": warnings,
        "invocation": {
            "data_path": str(args.data_path) if args.data_path is not None else None,
            "data_source": args.data_source,
            "source_dir": str(args.source_dir),
            "artifact_dir": str(args.artifact_dir),
            "expected_visible_gpus": args.expected_visible_gpus,
        },
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
            "implementation": platform.python_implementation(),
        },
        "host": {
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "allowed_cpus": len(os.sched_getaffinity(0)),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "limits": {
                "open_files": [_limit(value) for value in nofile],
                "locked_memory_bytes": [_limit(value) for value in memlock],
            },
        },
        "paths": {
            "data": data,
            "source": source,
            "artifacts": artifact,
            "shared_memory": shared_memory,
        },
        "torch": torch_info,
        "dependencies": {
            "dali_loader_evaluator": dali,
            "nvtx": nvtx,
            "nsys": nsys,
            "git": git,
        },
    }

    output = args.artifact_dir / "preflight.json"
    try:
        output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    except OSError as exc:
        print(f"preflight blocked: cannot write {output}: {_error(exc)}", file=sys.stderr)
        return 2

    print(f"preflight status={result['status']} json={output}")
    # argparse rejects anything below 1, so the count is either a truthy int or None.
    expected = args.expected_visible_gpus or "unspecified"
    print(f"gpu visible={len(torch_info['visible_devices'])} expected={expected}")
    print(f"dali={dali['available']} nvtx={nvtx['available']} nsys={nsys['available']}")
    for item in blockers + warnings:
        print(f"{item['code']}: {item['message']}", file=sys.stderr)
    return 1 if blockers else 0


if __name__ == "__main__":
    raise SystemExit(main())
