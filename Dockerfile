#########################################################################################
##  Stage 2: build DALI wheels on top of the dependencies image built in Stage 1
#########################################################################################
ARG DEPS_IMAGE_NAME
# clean builder without source code inside
FROM ${DEPS_IMAGE_NAME} as builder

ARG PYVER=2.7
ARG PYV=27

ENV PYVER=${PYVER} PYV=${PYV} PYTHONPATH=/opt/python/v

ENV PYBIN=${PYTHONPATH}/bin \
    PYLIB=${PYTHONPATH}/lib

ENV PATH=${PYBIN}:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/opt/dali/${DALI_BUILD_DIR}:${PYLIB}:${LD_LIBRARY_PATH}

RUN ln -s /opt/python/cp${PYV}* /opt/python/v

RUN pip install future numpy setuptools wheel && \
    rm -rf /root/.cache/pip/

RUN if [ ${PYV} != "37" ] ; then \
        pip install tensorflow-gpu==1.7                                && \
        pip install tensorflow-gpu==1.11   --target /tensorflow/1_11   && \
        pip install tensorflow-gpu==1.12   --target /tensorflow/1_12   && \
        pip install tensorflow-gpu==1.13.1 --target /tensorflow/1_13   && \
        pip install tensorflow-gpu         --target /tensorflow/latest;   \
    else                                                                  \
        # Older versions not supported on python 3.7
        pip install tensorflow-gpu; \
    fi && \
    rm -rf /root/.cache/pip/

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    ldconfig

WORKDIR /opt/dali

ARG CC
ARG CXX
ENV CC=${CC}
ENV CXX=${CXX}

ARG NVIDIA_DALI_BUILD_FLAVOR
ARG GIT_SHA
ARG DALI_TIMESTAMP

# Optional build arguments

ARG CMAKE_BUILD_TYPE=Release
ENV CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
ARG BUILD_TEST=ON
ENV BUILD_TEST=${BUILD_TEST}
ARG BUILD_BENCHMARK=ON
ENV BUILD_BENCHMARK=${BUILD_BENCHMARK}
ARG BUILD_NVTX=OFF
ENV BUILD_NVTX=${BUILD_NVTX}
ARG BUILD_PYTHON=ON
ENV BUILD_PYTHON=${BUILD_PYTHON}
ARG BUILD_LMDB=ON
ENV BUILD_LMDB=${BUILD_LMDB}
ARG BUILD_TENSORFLOW=ON
ENV BUILD_TENSORFLOW=${BUILD_TENSORFLOW}
ARG BUILD_JPEG_TURBO=ON
ENV BUILD_JPEG_TURBO=${BUILD_JPEG_TURBO}
ARG BUILD_NVJPEG=ON
ENV BUILD_NVJPEG=${BUILD_NVJPEG}
ARG BUILD_NVOF=ON
ENV BUILD_NVOF=${BUILD_NVOF}
ARG BUILD_NVDEC=ON
ENV BUILD_NVDEC=${BUILD_NVDEC}
ARG BUILD_NVML=ON
ENV BUILD_NVML=${BUILD_NVML}

ARG NVIDIA_BUILD_ID
ENV NVIDIA_BUILD_ID ${NVIDIA_BUILD_ID:-0}
RUN mkdir /wheelhouse && chmod 0777 /wheelhouse

FROM builder
COPY . .

ARG DALI_BUILD_DIR=build-docker-release
WORKDIR /opt/dali/${DALI_BUILD_DIR}

RUN /opt/dali/docker/build_helper.sh
