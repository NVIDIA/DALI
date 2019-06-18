#########################################################################################
##  Stage 2: build DALI wheels on top of the dependencies image built in Stage 1
#########################################################################################
ARG DEPS_IMAGE_NAME
# clean builder without source code inside
FROM ${DEPS_IMAGE_NAME} as builder

RUN ls -la /usr/local/cuda && cat /usr/local/cuda/version.txt

RUN ls -la /usr/local/cuda && cat /usr/local/cuda/version.txt && cd /usr/local/cuda/lib64/stubs/ && ln -s libcuda.so libcuda.so.1 && ls -la /usr/local/cuda && cat /usr/local/cuda/version.txt
RUN ls -la /usr/local/cuda && cat /usr/local/cuda/version.txt
