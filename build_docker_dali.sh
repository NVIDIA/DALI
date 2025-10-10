#!/usr/bin/env bash
set -euo pipefail

# -------- settings --------
DALI_DIR="${DALI_DIR:-/workspace/DALI}"
BUILD_DIR="${BUILD_DIR:-${DALI_DIR}/build}"
OUT_DIR="${OUT_DIR:-/out}"
PROTOBUF_ROOT="${PROTOBUF_ROOT:-/opt/protobuf-3.21.12}"   # matches the Dockerfile install path
PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"

# -------- sanity checks --------
if [[ ! -d "${DALI_DIR}/.git" ]]; then
  echo "❌ ${DALI_DIR} does not look like a git repo (is the volume mounted?)."
  exit 1
fi

# -------- make git repo 'safe' inside the container --------
# Trust the bind-mounted repo AND all submodules
git config --global --add safe.directory '*'

# -------- ensure submodules and LFS content are present --------
cd "${DALI_DIR}"

# Keep submodule URLs in sync with .gitmodules
git submodule sync --recursive

# Deinit all submodules to reset any partial/failed states, then re-init
git submodule deinit -f --all || true
git submodule update --init --recursive --jobs "$(nproc)"

# Initialize LFS for top-level repo and all submodules
git lfs install
git lfs fetch || true
git lfs checkout || true
git submodule foreach --recursive '
  git lfs install || true
  git lfs fetch || true
  git lfs checkout || true
'

# If any submodule commit is missing (e.g., "Unable to find current revision"),
# force-fetch its history and try again.
if ! git submodule status --recursive | awk "{print \$1}" | grep -vqE '^-'; then
  git submodule foreach --recursive '
    git fetch --tags origin || true
  '
  git submodule update --init --recursive --jobs "$(nproc)"
fi

# -------- protobuf env so CMake finds the right libraries --------
export CMAKE_PREFIX_PATH="${PROTOBUF_ROOT}:${CMAKE_PREFIX_PATH-}"
export LD_LIBRARY_PATH="${PROTOBUF_ROOT}/lib:${LD_LIBRARY_PATH-}"

# Make sure the dynamic loader can see libprotobuf/libprotoc
if [[ -w /etc/ld.so.conf.d ]]; then
  echo "${PROTOBUF_ROOT}/lib" >/etc/ld.so.conf.d/protobuf.conf
  ldconfig
fi

# -------- configure build --------
rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -Dprotobuf_MODULE_COMPATIBLE=ON \
  -DProtobuf_INCLUDE_DIR="${PROTOBUF_ROOT}/include" \
  -DProtobuf_LIBRARY="${PROTOBUF_ROOT}/lib/libprotobuf.so" \
  -DProtobuf_PROTOC_LIBRARY="${PROTOBUF_ROOT}/lib/libprotoc.so" \
  -DProtobuf_PROTOC_EXECUTABLE="${PROTOBUF_ROOT}/bin/protoc" \
  -DLIBTAR_LIBRARY=/usr/lib/x86_64-linux-gnu/libtar.so \
  -DPython3_EXECUTABLE="${PYTHON_BIN}"

# -------- build --------
make -j"$(nproc)"

# -------- export useful artifacts --------
mkdir -p "${OUT_DIR}"

# Core shared libs
find "${BUILD_DIR}/python/nvidia/dali" -maxdepth 1 -type f -name "*.so" -print -exec cp {} "${OUT_DIR}/" \; || true

# Unit test binaries (optional)
if compgen -G "${BUILD_DIR}/python/nvidia/dali/test/*.bin" > /dev/null; then
  mkdir -p "${OUT_DIR}/tests"
  cp -v "${BUILD_DIR}/python/nvidia/dali/test/"*.bin "${OUT_DIR}/tests/" || true
fi

echo "✅ Build complete."
echo "   Artifacts (if any) copied to: ${OUT_DIR}"
