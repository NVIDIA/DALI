#########################################################################################
##  Stage 2: build DALI wheels on top of the dependencies image built in Stage 1
#########################################################################################
ARG DEPS_IMAGE_NAME
# clean builder without source code inside
FROM ${DEPS_IMAGE_NAME} as builder


RUN ls -la /usr/local/cuda && cat /usr/local/cuda/version.txt

ARG PYVER=2.7
RUN ls -la /usr/local/cuda && cat /usr/local/cuda/version.txt
ARG PYV=27
RUN ls -la /usr/local/cuda && cat /usr/local/cuda/version.txt

ENV PYVER=${PYVER} PYV=${PYV} PYTHONPATH=/opt/python/v
RUN ls -la /usr/local/cuda && cat /usr/local/cuda/version.txt

ENV PYBIN=${PYTHONPATH}/bin \
    PYLIB=${PYTHONPATH}/lib
RUN ls -la /usr/local/cuda && cat /usr/local/cuda/version.txt

ENV PATH=${PYBIN}:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/opt/dali/${DALI_BUILD_DIR}:${PYLIB}:${LD_LIBRARY_PATH}
RUN ls -la /usr/local/cuda && cat /usr/local/cuda/version.txt

RUN ln -s /opt/python/cp${PYV}* /opt/python/v
RUN ls -la /usr/local/cuda && cat /usr/local/cuda/version.txt

RUN pip install future numpy setuptools wheel && \
    rm -rf /root/.cache/pip/
RUN ls -la /usr/local/cuda && cat /usr/local/cuda/version.txt

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    ldconfig
RUN ls -la /usr/local/cuda && cat /usr/local/cuda/version.txt

WORKDIR /opt/dali
RUN ls -la /usr/local/cuda && cat /usr/local/cuda/version.txt

ARG CC
ARG CXX
ENV CC=${CC}
ENV CXX=${CXX}
RUN ls -la /usr/local/cuda && cat /usr/local/cuda/version.txt
# Optional build arguments

ARG CMAKE_BUILD_TYPE
ENV CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
ARG BUILD_TEST
ENV BUILD_TEST=${BUILD_TEST}
ARG BUILD_BENCHMARK
ENV BUILD_BENCHMARK=${BUILD_BENCHMARK}
ARG BUILD_NVTX
ENV BUILD_NVTX=${BUILD_NVTX}
ARG BUILD_PYTHON
ENV BUILD_PYTHON=${BUILD_PYTHON}
ARG BUILD_LMDB
ENV BUILD_LMDB=${BUILD_LMDB}
ARG BUILD_TENSORFLOW
ENV BUILD_TENSORFLOW=${BUILD_TENSORFLOW}
RUN ls -la /usr/local/cuda && cat /usr/local/cuda/version.txt
ARG BUILD_JPEG_TURBO
ENV BUILD_JPEG_TURBO=${BUILD_JPEG_TURBO}
ARG BUILD_NVJPEG
ENV BUILD_NVJPEG=${BUILD_NVJPEG}
ARG BUILD_NVOF
ENV BUILD_NVOF=${BUILD_NVOF}
ARG BUILD_NVDEC
ENV BUILD_NVDEC=${BUILD_NVDEC}
ARG BUILD_NVML
ENV BUILD_NVML=${BUILD_NVML}
ARG NVIDIA_DALI_BUILD_FLAVOR
ENV NVIDIA_DALI_BUILD_FLAVOR=${NVIDIA_DALI_BUILD_FLAVOR}
ARG GIT_SHA
ENV GIT_SHA=${GIT_SHA}
ARG DALI_TIMESTAMP
ENV DALI_TIMESTAMP=${DALI_TIMESTAMP}
RUN ls -la /usr/local/cuda && cat /usr/local/cuda/version.txt

ARG NVIDIA_BUILD_ID
ENV NVIDIA_BUILD_ID ${NVIDIA_BUILD_ID:-0}
RUN mkdir /wheelhouse && chmod 0777 /wheelhouse
RUN ls -la /usr/local/cuda && cat /usr/local/cuda/version.txt

FROM builder
RUN ls -la /usr/local/cuda && cat /usr/local/cuda/version.txt
COPY . .
RUN ls -la /usr/local/cuda && cat /usr/local/cuda/version.txt

ARG DALI_BUILD_DIR=build-docker-release
RUN ls -la /usr/local/cuda && cat /usr/local/cuda/version.txt
WORKDIR /opt/dali/${DALI_BUILD_DIR}
RUN ls -la /usr/local/cuda && cat /usr/local/cuda/version.txt


RUN /opt/dali/docker/build_helper.sh
