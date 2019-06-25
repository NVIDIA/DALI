FROM tensorflow/tensorflow:custom-op-gpu
ARG TF_VERSION=1.14.0
ENV TF_VERSION=${TF_VERSION}

RUN pip install tensorflow-gpu==${TF_VERSION} && \
    rm -rf /root/.cache/pip/
