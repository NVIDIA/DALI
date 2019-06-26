FROM tensorflow/tensorflow:custom-op-gpu
ARG TF_VERSION=1.14.0
ENV TF_VERSION=${TF_VERSION}

RUN pip install tensorflow-gpu==${TF_VERSION} && \
    rm -rf /root/.cache/pip/

RUN mkdir /dali_tf_plugins && chmod 0777 /dali_tf_plugins

COPY . .

RUN pip install whl/*.whl

ENTRYPOINT /bin/bash build_in_custom_op_docker.sh