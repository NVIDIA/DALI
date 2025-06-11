#!/usr/bin/env bash

ARCH=${ARCH:-$(uname -p)}
ARTIFACTORY_USER=${ARTIFACTORY_USER:-""}
ARTIFACTORY_API_TOKEN=${ARTIFACTORY_API_TOKEN:-""}

if [ -z "${ARTIFACTORY_USER}" ] || [ -z "${ARTIFACTORY_API_TOKEN}" ]; then
    echo "ARTIFACTORY_USER and ARTIFACTORY_API_TOKEN must be set"
    exit 1
fi

mkdir -p /extra-pip-packages
cd /extra-pip-packages

if [ "${ARCH}" = "tegra" ]; then
    curl -u ${ARTIFACTORY_USER}:${ARTIFACTORY_API_TOKEN} \
        https://urm.nvidia.com/artifactory/sw-cuda-math-nvtiff-pypi-local/release-build/nvidia-nvtiff/r13/0.5.1.69/nvidia_nvtiff_tegra_cu13-0.5.1.69-py3-none-manylinux2014_aarch64.whl -O
    curl -u ${ARTIFACTORY_USER}:${ARTIFACTORY_API_TOKEN} \
        https://urm.nvidia.com/artifactory/sw-cuda-math-nvjpeg2k-pypi-local/for-installer/nvidia-nvjpeg2k/r13/0.9.0.42/nvidia_nvjpeg2k_tegra_cu13-0.9.0.42-py3-none-manylinux2014_aarch64.whl -O
    curl -u ${ARTIFACTORY_USER}:${ARTIFACTORY_API_TOKEN} \
        https://urm-sc.nvidia.com/artifactory/sw-cuda-math-nvimagecodec-pypi-local/cicd/main/L0_MergeRequest_13/r13.0/nvimgcodec/linux-aarch64/0.6.0.135/nvidia_nvimgcodec_tegra_cu13-0.6.0.135-py3-none-manylinux2014_aarch64.whl -O
else
    curl -u ${ARTIFACTORY_USER}:${ARTIFACTORY_API_TOKEN} \
        https://urm.nvidia.com/artifactory/sw-cuda-math-nvtiff-pypi-local/release-build/nvidia-nvtiff/r13/0.5.1.69/nvidia_nvtiff_cu13-0.5.1.69-py3-none-manylinux2014_${ARCH}.whl -O
    curl -u ${ARTIFACTORY_USER}:${ARTIFACTORY_API_TOKEN} \
        https://urm.nvidia.com/artifactory/sw-cuda-math-nvjpeg2k-pypi-local/for-installer/nvidia-nvjpeg2k/r13/0.9.0.42/nvidia_nvjpeg2k_cu13-0.9.0.42-py3-none-manylinux2014_${ARCH}.whl -O
    curl -u ${ARTIFACTORY_USER}:${ARTIFACTORY_API_TOKEN} \
        https://urm-sc.nvidia.com/artifactory/sw-cuda-math-nvimagecodec-pypi-local/cicd/main/L0_MergeRequest_13/135/nvidia_nvimgcodec_cu13-0.6.0.135-py3-none-manylinux2014_${ARCH}.whl -O
fi



