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

cd /opt/
git clone https://github.com/conda-forge/nvidia-dali-python-feedstock

cd nvidia-dali-python-feedstock
if [ -n "${DALI_FEEDSTOCK_SHA}" ]; then
  git checkout ${DALI_FEEDSTOCK_SHA}
fi

echo "Using nvidia-dali-python-feedstock at $(git rev-parse HEAD)"

# Use the local DALI checkout instead of downloading the release tarball.
RECIPE=recipe/recipe.yaml
sed -i 's|  - url: https://github.com/NVIDIA/DALI/archive/refs/tags/v${{ version }}.tar.gz|  - path: /opt/dali|' "${RECIPE}"
grep -q 'path: /opt/dali' "${RECIPE}" || { echo "Recipe patch failed: path: /opt/dali not found" >&2; exit 1; }
grep -q 'github.com/NVIDIA/DALI/archive' "${RECIPE}" && \
  { echo "Recipe patch failed: DALI tarball URL still present" >&2; exit 1; }

sed -i '/path: \/opt\/dali/a\    use_gitignore: false' "${RECIPE}"
grep -q 'use_gitignore: false' "${RECIPE}" || \
  { echo "Recipe patch failed: use_gitignore: false not found" >&2; exit 1; }

# ToDo remove when feedstock is updated to DALI 2.2
# Local checkout already contains upstream changes; drop feedstock-only patches until it moves to
# DALI 2.2
sed -i '/    patches:/,/^  - url:/{ /^  - url:/!d; }' "${RECIPE}"
grep -q 'patches/0001-BLD' "${RECIPE}" && \
  { echo "Recipe patch failed: DALI patches still present" >&2; exit 1; }

sed -i '0,/^    sha256: /{/^    sha256: /d;}' "${RECIPE}"

# The local DALI checkout already has initialized submodules. Feedstock adds these sources only
# because GitHub release tarballs do not contain submodule contents.
sed -i '\|https://github.com/cocodataset/cocoapi/archive/|,+2d' "${RECIPE}"
sed -i '\|https://github.com/JanuszL/ffts/archive/|,+2d' "${RECIPE}"
grep -q 'target_directory: third_party/cocoapi' "${RECIPE}" && \
  { echo "Recipe patch failed: cocoapi source still present" >&2; exit 1; }
grep -q 'target_directory: third_party/ffts' "${RECIPE}" && \
  { echo "Recipe patch failed: ffts source still present" >&2; exit 1; }

BUILD_SCRIPT=recipe/build.sh
sed -i '/^ln -sf \$PREFIX\/include\/boost /i\rm -rf third_party/boost/preprocessor/include/boost' \
  "${BUILD_SCRIPT}"
sed -i '/^ln -sf \$PREFIX\/include\/dlpack /i\rm -rf third_party/dlpack/include/dlpack' \
  "${BUILD_SCRIPT}"
sed -i '/^ln -sf \$PREFIX\/include\/cute /i\rm -rf third_party/cutlass/include/cute' \
  "${BUILD_SCRIPT}"
sed -i '/^ln -sf \$PREFIX\/include\/cutlass /i\rm -rf third_party/cutlass/include/cutlass' \
  "${BUILD_SCRIPT}"
grep -q 'rm -rf third_party/boost/preprocessor/include/boost' "${BUILD_SCRIPT}" || \
  { echo "build.sh patch failed: boost symlink cleanup not found" >&2; exit 1; }

sed -i '/^cmake \${CMAKE_ARGS} \\$/,/^\$SRC_DIR$/ s/^  -GNinja \\$/  -GNinja \\\n  -DBUILD_FOR_CONDA=ON \\/' \
  "${BUILD_SCRIPT}"
grep -q 'BUILD_FOR_CONDA=ON' "${BUILD_SCRIPT}" || \
  { echo "build.sh patch failed: BUILD_FOR_CONDA=ON not found" >&2; exit 1; }
# End of ToDo

rattler-build build --recipe $(pwd)/recipe -m  $(pwd)/.ci_support/${METAL_YAML} \
                    --target-platform linux-64 --extra-meta flow_run_id=${NVIDIA_BUILD_ID:-0} \
                    --extra-meta remote_url= --extra-meta sha=${GIT_SHA}

# Copying the artifacts from conda prefix
mkdir -p artifacts
cp output/linux-64/*.conda artifacts
