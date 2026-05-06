# Point libnvimgcodec at the conda-forge extension modules.
# The default search path baked into libnvimgcodec.so points at /opt/nvidia/nvimgcodec_cuda,
# which doesn't exist in a conda-forge install — extensions are installed under
# $CONDA_PREFIX/lib/extensions instead.
if [ -z "${NVIMGCODEC_EXTENSIONS_PATH:-}" ]; then
    export _DALI_SET_NVIMGCODEC_EXTENSIONS_PATH=1
    export NVIMGCODEC_EXTENSIONS_PATH="${CONDA_PREFIX}/lib/extensions"
fi
