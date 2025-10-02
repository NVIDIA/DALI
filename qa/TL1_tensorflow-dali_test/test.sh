#!/bin/bash -e
target_dir=./docs/examples/use_cases/tensorflow/resnet-n

do_once() {
    if [ $($topdir/qa/setup_packages.py -n -u tensorflow-gpu --cuda ${CUDA_VERSION}) = -1 ]; then
        exit 0
    fi
    mkdir -p idx-files/

    NUM_GPUS=$(nvidia-smi -L | wc -l)

    CUDA_VERSION=$(echo $(nvcc --version) | sed 's/.*\(release \)\([0-9]\+\)\.\([0-9]\+\).*/\2\3/')

    idx=$($topdir/qa/setup_packages.py -n -u tensorflow-gpu --cuda ${CUDA_VERSION})
    install_pip_pkg "pip install $($topdir/qa/setup_packages.py -i $idx -u tensorflow-gpu --cuda ${CUDA_VERSION}) -f /pip-packages"

    # The package name can be nvidia_dali_tf_plugin,  nvidia_dali_tf_plugin-weekly or  nvidia_dali_tf_plugin-nightly
    pip uninstall -y `pip list | grep nvidia_dali_tf_plugin | cut -d " " -f1` || true

    pip install /opt/dali/nvidia_dali_tf_plugin*.tar.gz

    export PATH=$PATH:/usr/local/mpi/bin
    # MPI might be present in CUDA 10 image already so no need to build it if that is the case
    if ! [ -x "$(command -v mpicxx)" ]; then
        apt-get update && apt-get install -y wget
        OPENMPI_VERSION=3.0.0
        wget -q -O - https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-${OPENMPI_VERSION}.tar.gz | tar -xzf -
        cd openmpi-${OPENMPI_VERSION}
        ./configure --enable-orterun-prefix-by-default --prefix=/usr/local/mpi --disable-getpwuid -with-cma=no
        make -j"$(nproc)" install
        cd .. && rm -rf openmpi-${OPENMPI_VERSION}
        echo "/usr/local/mpi/lib" >> /etc/ld.so.conf.d/openmpi.conf && ldconfig

        /bin/echo -e '#!/bin/bash'\
        '\ncat <<EOF'\
        '\n======================================================================'\
        '\nTo run a multi-node job, install an ssh client and clear plm_rsh_agent'\
        '\nin '/usr/local/mpi/etc/openmpi-mca-params.conf'.'\
        '\n======================================================================'\
        '\nEOF'\
        '\nexit 1' >> /usr/local/mpi/bin/rsh_warn.sh
        chmod +x /usr/local/mpi/bin/rsh_warn.sh
        echo "plm_rsh_agent = /usr/local/mpi/bin/rsh_warn.sh" >> /usr/local/mpi/etc/openmpi-mca-params.conf
    fi

    apt-get update && apt-get install -y cmake
    export HOROVOD_GPU_ALLREDUCE=NCCL
    export HOROVOD_NCCL_INCLUDE=/usr/include
    export HOROVOD_NCCL_LIB=/usr/lib/x86_64-linux-gnu
    export HOROVOD_NCCL_LINK=SHARED
    export HOROVOD_WITHOUT_PYTORCH=1
    # it addresses the issue with third_party/gloo, which is a part of horovod dependency,
    # requiring too old version of cmake
    export CMAKE_POLICY_VERSION_MINIMUM=3.5
    # patch and install horovod
    git clone https://github.com/abseil/abseil-cpp.git --depth 1 -b lts_2025_01_27 && \
    cd abseil-cpp && mkdir build && cd build && \
    CFLAGS="-fPIC" CXXFLAGS="-fPIC -std=c++17" cmake ../ && make -j$(nproc) && make install && cd ../.. && rm -rf abseil-cpp && \
    pip download horovod==0.28.1 && tar -xf horovod-0.28.1.tar.gz  && cd horovod-0.28.1 && \
    sed -i "s/tensorflow\/compiler\/xla\/client\/xla_builder\.h/xla\/hlo\/builder\/xla_builder\.h/" horovod/tensorflow/xla_mpi_ops.cc && \
    sed -i "s/::xla::StatusOr/absl::StatusOr/" horovod/tensorflow/xla_mpi_ops.cc && \
    echo "find_package(absl REQUIRED)" >> horovod/tensorflow/CMakeLists.txt && \
    echo "target_link_libraries(\${TF_TARGET_LIB} absl::log)" >> horovod/tensorflow/CMakeLists.txt && \
    pip install . && cd .. && rm -rf horovod*

    for file in $(ls /data/imagenet/train-val-tfrecord-small);
    do
        python ../../../../../tools/tfrecord2idx /data/imagenet/train-val-tfrecord-small/${file} \
            idx-files/${file}.idx &
    done
    wait

    # OMPI_MCA_blt is needed if running in a container w/o cap SYS_PRACE
    # More info: https://github.com/radiasoft/devops/issues/132
    export OMPI_MCA_blt=self,sm,tcp
}

test_body() {
    # for the compatibility with the old keras API
    export TF_USE_LEGACY_KERAS=1
    # test code
    mpiexec --allow-run-as-root --bind-to none -np ${NUM_GPUS} \
        python -u resnet.py \
        --data_dir=/data/imagenet/train-val-tfrecord-small --data_idx_dir=idx-files/ \
        --precision=fp16 --num_iter=100  --iter_unit=batch --display_every=50 \
        --batch=64 --use_xla --dali_mode="GPU" --log_dir=./
}

pushd ../..
source ./qa/test_template.sh
popd
