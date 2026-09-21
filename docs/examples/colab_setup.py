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

"""Prepare a Google Colab (or any other Jupyter) runtime for running DALI example notebooks.

The script installs the DALI wheel matching the CUDA version available in the runtime, fetches
the `DALI_extra <https://github.com/NVIDIA/DALI_extra>`_ test data at the revision expected by
the notebooks and exports ``DALI_EXTRA_PATH`` so that the notebooks can locate it.

Run it from a notebook cell so that the environment variable is set in the kernel process::

    !curl -sSL https://raw.githubusercontent.com/NVIDIA/DALI/main/docs/examples/colab_setup.py \\
        -o colab_setup.py
    %run colab_setup.py

It can also be run as a regular script (``python colab_setup.py``); in that case it prints the
``export DALI_EXTRA_PATH=...`` command to run afterwards.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import urllib.request

DALI_REPO = "NVIDIA/DALI"
DALI_EXTRA_URL = "https://github.com/NVIDIA/DALI_extra.git"
RELEASE_INDEX_URL = "https://pypi.nvidia.com"
NIGHTLY_INDEX_URL = (
    "https://developer.download.nvidia.com/compute/redist/nightly"
)
SUPPORTED_CUDA_MAJORS = (12, 13)


def _run(cmd, **kwargs):
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, **kwargs)


def _detect_cuda_major():
    """Detect the CUDA major version supported by the driver via ``nvidia-smi``."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi"], text=True, stderr=subprocess.STDOUT
        )
    except (OSError, subprocess.CalledProcessError) as e:
        raise RuntimeError(
            "nvidia-smi is not available - DALI requires a GPU runtime. In Colab select "
            "Runtime -> Change runtime type -> Hardware accelerator: GPU."
        ) from e
    match = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    if not match:
        raise RuntimeError(
            f"Could not determine the CUDA version from nvidia-smi output:\n{out}"
        )
    return int(match.group(1))


def _pick_cuda_variant(cuda_major):
    for supported in reversed(SUPPORTED_CUDA_MAJORS):
        if cuda_major >= supported:
            return supported
    raise RuntimeError(
        f"CUDA {cuda_major} driver is too old for DALI, one of CUDA {SUPPORTED_CUDA_MAJORS} "
        "capable drivers is required."
    )


def install_dali(cuda_major=None, version=None, nightly=False):
    """Install the DALI wheel for the given CUDA major version (auto-detected when None)."""
    if cuda_major is None:
        cuda_major = _pick_cuda_variant(_detect_cuda_major())
    if nightly:
        package = f"nvidia-dali-nightly-cuda{cuda_major}0"
        index_url = NIGHTLY_INDEX_URL
    else:
        package = f"nvidia-dali-cuda{cuda_major}0"
        index_url = RELEASE_INDEX_URL
    if version:
        package = f"{package}=={version}"
    _run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--quiet",
            "--extra-index-url",
            index_url,
            "--upgrade",
            package,
        ]
    )
    return package


def fetch_dali_extra_version(ref):
    """Read ``DALI_EXTRA_VERSION`` from the DALI repository at the given git ref."""
    url = f"https://raw.githubusercontent.com/{DALI_REPO}/{ref}/DALI_EXTRA_VERSION"
    with urllib.request.urlopen(url) as response:
        return response.read().decode("ascii").strip()


def _ensure_git_lfs():
    if shutil.which("git-lfs"):
        return
    if shutil.which("apt-get"):
        _run(["apt-get", "install", "-y", "-qq", "git-lfs"])
    else:
        raise RuntimeError(
            "git-lfs is required to fetch DALI_extra, see https://git-lfs.com for installation."
        )


def setup_dali_extra(path, revision):
    """Clone DALI_extra into ``path`` (if missing) and check out ``revision``."""
    _ensure_git_lfs()
    _run(["git", "lfs", "install", "--skip-repo"])
    if not os.path.isdir(os.path.join(path, ".git")):
        os.makedirs(path, exist_ok=True)
        _run(["git", "init", "-q", path])
        _run(["git", "-C", path, "remote", "add", "origin", DALI_EXTRA_URL])
    _run(["git", "-C", path, "fetch", "-q", "--depth", "1", "origin", revision])
    _run(["git", "-C", path, "checkout", "-q", "FETCH_HEAD"])
    _run(["git", "-C", path, "lfs", "pull"])
    return os.path.abspath(path)


def _running_in_ipython():
    try:
        get_ipython  # noqa: F821 - defined by IPython/Jupyter kernels
        return True
    except NameError:
        return False


def _default_dali_extra_path():
    # /content is the working directory of Colab runtimes
    base = "/content" if os.path.isdir("/content") else os.getcwd()
    return os.path.join(base, "DALI_extra")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--ref",
        default="main",
        help="DALI git ref (branch, tag or commit) the notebooks come from; used to pick the "
        "matching DALI_extra revision (default: main)",
    )
    parser.add_argument(
        "--dali-version",
        default=None,
        help="DALI wheel version to install (default: the latest available)",
    )
    parser.add_argument(
        "--nightly",
        action="store_true",
        help="install the nightly DALI build instead of the latest release",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        choices=SUPPORTED_CUDA_MAJORS,
        default=None,
        help="CUDA major version of the DALI wheel (default: detected with nvidia-smi)",
    )
    parser.add_argument(
        "--skip-dali",
        action="store_true",
        help="do not install DALI (e.g. when it is already installed)",
    )
    parser.add_argument(
        "--dali-extra-path",
        default=_default_dali_extra_path(),
        help="where to clone DALI_extra (default: %(default)s)",
    )
    parser.add_argument(
        "--dali-extra-version",
        default=None,
        help="DALI_extra revision to check out (default: DALI_EXTRA_VERSION at --ref)",
    )
    parser.add_argument(
        "--skip-dali-extra",
        action="store_true",
        help="do not fetch DALI_extra test data",
    )
    args = parser.parse_args(argv)

    if not args.skip_dali:
        package = install_dali(args.cuda, args.dali_version, args.nightly)
        print(f"Installed {package}")

    if not args.skip_dali_extra:
        revision = args.dali_extra_version or fetch_dali_extra_version(args.ref)
        dali_extra_path = setup_dali_extra(args.dali_extra_path, revision)
        os.environ["DALI_EXTRA_PATH"] = dali_extra_path
        print(f"DALI_extra {revision} is available at {dali_extra_path}")
        if _running_in_ipython():
            print(
                f"DALI_EXTRA_PATH={dali_extra_path} exported to this notebook kernel."
            )
        else:
            print(
                f"Run `export DALI_EXTRA_PATH={dali_extra_path}` before starting Jupyter."
            )


if __name__ == "__main__":
    main()
