#########################################################################################
##  Stage 2: build DALI wheels on top of the dependencies image built in Stage 1
#########################################################################################
ARG DEPS_IMAGE_NAME
FROM ${DEPS_IMAGE_NAME}

ARG PYVER=2.7
ARG PYV=27

ENV PYVER=${PYVER} PYV=${PYV} PYTHONPATH=/opt/python/v

ENV PYBIN=${PYTHONPATH}/bin \
    PYLIB=${PYTHONPATH}/lib

ENV PATH=${PYBIN}:${PATH} \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64/stubs:/opt/dali/build:${PYLIB}:${LD_LIBRARY_PATH}

RUN ln -s /opt/python/cp${PYV}* /opt/python/v

RUN pip install future numpy setuptools wheel tensorflow-gpu==1.7 && \
    rm -rf /root/.cache/pip/

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 && \
    ldconfig

WORKDIR /opt/dali

COPY . .

WORKDIR /opt/dali/build

RUN LD_LIBRARY_PATH="${PWD}:${LD_LIBRARY_PATH}" && \
    cmake ../ -DCMAKE_INSTALL_PREFIX=. \
          -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON -DBUILD_PYTHON=ON \
          -DBUILD_LMDB=ON -DBUILD_TENSORFLOW=ON && \
    make -j"$(grep ^processor /proc/cpuinfo | wc -l)"

ARG NVIDIA_BUILD_ID
ENV NVIDIA_BUILD_ID ${NVIDIA_BUILD_ID:-0}

RUN pip wheel -v dali/python \
        --build-option --python-tag=$(basename /opt/python/cp${PYV}-*) \
        --build-option --plat-name=manylinux1_x86_64 \
        --build-option --build-number=${NVIDIA_BUILD_ID} && \
    ../dali/python/bundle-wheel.sh nvidia_dali-*.whl
