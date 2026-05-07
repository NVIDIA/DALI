# Point libnvimgcodec at the conda-forge extension modules.
# libnvimgcodec resolves its default extensions path dynamically via dladdr:
# with libnvimgcodec.so installed at $CONDA_PREFIX/lib it strips the "lib"
# component and probes $CONDA_PREFIX/extensions, but the conda-forge feedstock
# installs the extension shared libraries under $CONDA_PREFIX/lib/extensions,
# so the computed default lands one directory off and every decode fails with
# NVIMGCODEC_PROCESSING_STATUS_*. Workaround: pin NVIMGCODEC_EXTENSIONS_PATH
# from the activate hook. Drop this once libnvimgcodec's default-path
# resolution (or the feedstock install layout) is fixed.
if [ -z "${NVIMGCODEC_EXTENSIONS_PATH:-}" ]; then
    export _DALI_SET_NVIMGCODEC_EXTENSIONS_PATH=1
    export NVIMGCODEC_EXTENSIONS_PATH="${CONDA_PREFIX}/lib/extensions"
fi
