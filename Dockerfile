#########################################################################################
##  Stage 2: build DALI wheels on top of the dependencies image built in Stage 1
#########################################################################################
ARG DEPS_IMAGE_NAME
FROM ${DEPS_IMAGE_NAME}

ARG PYVER=2.7
ARG PYV=27
ARG DALI_BUILD_DIR=build-docker-release

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

COPY . .

WORKDIR /opt/dali/${DALI_BUILD_DIR}

ARG CC
ARG CXX
ENV CC=${CC}
ENV CXX=${CXX}

RUN LD_LIBRARY_PATH="${PWD}:${LD_LIBRARY_PATH}" && \
    cmake ../ -DCMAKE_INSTALL_PREFIX=. \
          -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON -DBUILD_PYTHON=ON \
          -DBUILD_LMDB=ON -DBUILD_TENSORFLOW=ON -DWERROR=ON && \
    make -j"$(grep ^processor /proc/cpuinfo | wc -l)"

ARG NVIDIA_BUILD_ID
ENV NVIDIA_BUILD_ID ${NVIDIA_BUILD_ID:-0}

RUN pip wheel -v dali/python \
        --build-option --python-tag=$(basename /opt/python/cp${PYV}-*) \
        --build-option --plat-name=manylinux1_x86_64 \
        --build-option --build-number=${NVIDIA_BUILD_ID} && \
    ../dali/python/bundle-wheel.sh nvidia_dali[_-]*.whl && \
    UNZIP_PATH="$(mktemp -d)" && \
    unzip /wheelhouse/nvidia_dali*.whl -d $UNZIP_PATH && \
    python ../tools/test_bundled_libs.py $(find $UNZIP_PATH -iname *.so* | tr '\n' ' ') && \
    rm -rf $UNZIP_PATH

RUN pushd dali/python/tf_plugin/ && \
    python setup.py sdist && \
    mv dist/nvidia-dali-tf-plugin*.tar.gz /wheelhouse/ && \
    popd
