#!/bin/bash

set -o xtrace
set -e

source /root/miniconda3/bin/activate

if [ -z "${CUDA_VERSION_MAJOR}" ]; then
  echo "CUDA_VERSION_MAJOR must be set (e.g. 12 or 13)" >&2
  exit 1
fi
if [ "${CUDA_VERSION_MAJOR}" -eq 13 ]; then
  export METAL_YAML=linux_64_c_stdlib_version2.28cuda_compiler_version13.0.yaml
else
  export METAL_YAML=linux_64_c_stdlib_version2.17cuda_compiler_version12.9.yaml
fi

export DALI_VERSION="$(tr -d '\r\n' < /opt/dali/VERSION)"
export NVIDIA_DALI_BUILD_FLAVOR="${NVIDIA_DALI_BUILD_FLAVOR:-}"
export DALI_TIMESTAMP="${DALI_TIMESTAMP:-}"
export NVIDIA_BUILD_ID="${NVIDIA_BUILD_ID:-0}"

if [[ ! "${DALI_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+[[:alnum:].]*$ ]]; then
  echo "Invalid DALI version in /opt/dali/VERSION: ${DALI_VERSION}" >&2
  exit 1
fi
if [[ -n "${NVIDIA_DALI_BUILD_FLAVOR}" && ! "${DALI_TIMESTAMP}" =~ ^[0-9]{8}$ ]]; then
  echo "DALI_TIMESTAMP must use YYYYMMDD format when NVIDIA_DALI_BUILD_FLAVOR is set" >&2
  exit 1
fi
if [[ ! "${NVIDIA_BUILD_ID}" =~ ^[0-9]+$ ]]; then
  echo "NVIDIA_BUILD_ID must be a non-negative integer" >&2
  exit 1
fi

cd /opt/
git clone https://github.com/conda-forge/nvidia-dali-python-feedstock

cd nvidia-dali-python-feedstock
if [ -n "${DALI_FEEDSTOCK_SHA}" ]; then
  git checkout ${DALI_FEEDSTOCK_SHA}
fi

echo "Using nvidia-dali-python-feedstock at $(git rev-parse HEAD)"

# Adapt the current feedstock for building from the local DALI checkout. The patch removes
# release-tarball-only sources and patches, clears vendored include directories before the
# feedstock creates conda include symlinks, enables DALI's conda build configuration, and
# documents the nvCOMP 5.2 requirement.
git apply --check /opt/dali/conda/nvidia-dali-python-feedstock.patch
git apply /opt/dali/conda/nvidia-dali-python-feedstock.patch

# Use the local DALI checkout instead of downloading the release tarball.
RECIPE=recipe/recipe.yaml
sed -i 's|  - url: https://github.com/NVIDIA/DALI/archive/refs/tags/v${{ version }}.tar.gz|  - path: /opt/dali|' "${RECIPE}"
grep -q 'path: /opt/dali' "${RECIPE}" || { echo "Recipe patch failed: path: /opt/dali not found" >&2; exit 1; }
grep -q 'github.com/NVIDIA/DALI/archive' "${RECIPE}" && \
  { echo "Recipe patch failed: DALI tarball URL still present" >&2; exit 1; }

sed -i '/path: \/opt\/dali/a\    use_gitignore: false' "${RECIPE}"
grep -q 'use_gitignore: false' "${RECIPE}" || \
  { echo "Recipe patch failed: use_gitignore: false not found" >&2; exit 1; }

rattler-build build --recipe $(pwd)/recipe -m  $(pwd)/.ci_support/${METAL_YAML} \
                    --target-platform linux-64 --extra-meta flow_run_id=${NVIDIA_BUILD_ID:-0} \
                    --extra-meta remote_url= --extra-meta sha=${GIT_SHA}

# Copying the artifacts from conda prefix
mkdir -p /opt/dali/conda/artifacts/
cp output/linux-64/*.conda /opt/dali/conda/artifacts/
