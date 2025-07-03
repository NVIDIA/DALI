#!/bin/bash -e

pip_packages=''

do_once() {
    apt-get update && apt-get -y install wget
    wget https://github.com/Kitware/CMake/releases/download/v3.25.2/cmake-3.25.2-Linux-x86_64.sh
    bash cmake-*.sh --skip-license --prefix=/usr
    rm cmake-*.sh
    # for stub generation
    pip install clang==14.0
    pip install libclang==14.0.1
}

test_body() {
    export DALI_DIR=$PWD

    ## 1. Check that add_subdirectories(dali) works and are able to build the targets on the fly
    export TMP_DIR1="$(mktemp -d)"
    cp ${DALI_DIR}/qa/TL1_nodeps_build/CMakeLists_submodule.txt ${TMP_DIR1}/CMakeLists.txt
    cp ${DALI_DIR}/qa/TL1_nodeps_build/main.cc ${TMP_DIR1}
    pushd ${TMP_DIR1}
    ln -s ${DALI_DIR} .
    mkdir build
    pushd build
    cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc ..
    make -j
    ./nodeps_test
    popd
    popd
    rm -rf ${TMP_DIR1}

    # TODO: Fix make install!
    # ## 2. Build DALI core and kernel as shared-object libraries and install them in a stand-alone directory
    # export TMP_DIR2="$(mktemp -d)"
    # export TMP_DIR3="$(mktemp -d)"
    # export TMP_DIR4="$(mktemp -d)"
    # pushd ${TMP_DIR2}
    # cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DBUILD_DALI_NODEPS=ON -DSTATIC_LIBS=OFF \
    #       -DCMAKE_INSTALL_PREFIX=${TMP_DIR3} ${DALI_DIR}
    # make -j install dali_core dali_kernels
    # popd
    # rm -rf ${TMP_DIR2}

    # ## 3. Use DALI from the installation directory from step 2.
    # cd ${TMP_DIR4}
    # cp ${DALI_DIR}/qa/TL1_nodeps_build/CMakeLists_system.txt ${TMP_DIR4}/CMakeLists.txt
    # cp ${DALI_DIR}/qa/TL1_nodeps_build/main.cc ${TMP_DIR4}
    # pushd ${TMP_DIR4}
    # mkdir build
    # pushd build
    # cmake -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DCMAKE_CXX_FLAGS=-isystem\ ${TMP_DIR3}/include\ -L${TMP_DIR3}/lib ..
    # make -j
    # LD_LIBRARY_PATH=${TMP_DIR3}/lib ./nodeps_test
    # popd
    # popd
    # rm -rf ${TMP_DIR3}
    # rm -rf ${TMP_DIR4}
}

pushd ../..
source ./qa/test_template.sh
popd
