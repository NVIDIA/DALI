FROM tensorflow/tensorflow:custom-op-gpu
ARG TF_VERSION=1.14.0
ENV TF_VERSION=${TF_VERSION}

RUN pip install tensorflow-gpu==${TF_VERSION} && \
    rm -rf /root/.cache/pip/

RUN mkdir /dali_tf_plugins && chmod 0777 /dali_tf_plugins

COPY . .

RUN pip install whl/*.whl

RUN set -o xtrace && \
    SRCS="daliop.cc" && \
    SUFFIX=$(echo $TF_VERSION | sed 's/\([0-9]\+\)\.\([0-9]\+\).*/\1_\2/') && \
    LIB_NAME="libdali_tf_${SUFFIX}.so" && echo $LIB_NAME && \
    INCL_DIRS="-I/usr/local/cuda/targets/x86_64-linux/include/" &&  \
    TF_CFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') &&  \
    TF_LFLAGS=$(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') &&  \
    PYTHON_DIST_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])") && \
    DALI_TOPDIR="${PYTHON_DIST_PACKAGES}/nvidia/dali" && \
    DALI_CFLAGS="-I${DALI_TOPDIR}/include -D_GLIBCXX_USE_CXX11_ABI=0" && \
    DALI_LFLAGS="-L${DALI_TOPDIR} -ldali" && \
    g++ -std=c++11 -O2 -shared -fPIC ${SRCS} -o /dali_tf_plugins/${LIB_NAME} ${INCL_DIRS} ${TF_CFLAGS} ${TF_LFLAGS} ${DALI_CFLAGS} ${DALI_LFLAGS}
