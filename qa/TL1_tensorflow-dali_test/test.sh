#!/bin/bash -e
pip_packages=""
target_dir=./docs/examples/use_cases/tensorflow/resnet-n

do_once() {
    mkdir -p idx-files/

    NUM_GPUS=$(nvidia-smi -L | wc -l)

    CUDA_VERSION=$(echo $(ls /usr/local/cuda/lib64/libcudart.so*)  | sed 's/.*\.\([0-9]\+\)\.\([0-9]\+\)\.\([0-9]\+\)/\1\2/')

    # install any for CUDA 9 and the second for CUDA 10, 1.14 doesn't work so well with horovod
    pip install $($topdir/qa/setup_packages.py -i 1 -u tensorflow-gpu --cuda ${CUDA_VERSION}) -f /pip-packages

    pip uninstall -y nvidia-dali-tf-plugin || true
    pip install /opt/dali/nvidia-dali-tf-plugin*.tar.gz

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

    export HOROVOD_GPU_ALLREDUCE=NCCL
    export HOROVOD_NCCL_INCLUDE=/usr/include
    export HOROVOD_NCCL_LIB=/usr/lib/x86_64-linux-gnu
    export HOROVOD_NCCL_LINK=SHARED
    export HOROVOD_WITHOUT_PYTORCH=1
    pip install horovod==0.15.1

    for file in $(ls /data/imagenet/train-val-tfrecord-480-subset);
    do
        python ../../../../../tools/tfrecord2idx /data/imagenet/train-val-tfrecord-480-subset/${file} \
            idx-files/${file}.idx;
    done

    # OMPI_MCA_blt is needed if running in a container w/o cap SYS_PRACE
    # More info: https://github.com/radiasoft/devops/issues/132
    export OMPI_MCA_blt=self,sm,tcp
}

test_body() {
    # test code
    mpiexec --allow-run-as-root --bind-to socket -np ${NUM_GPUS} \
        python -u resnet.py --layers=18 \
        --data_dir=/data/imagenet/train-val-tfrecord-480-subset --data_idx_dir=idx-files/ \
        --precision=fp16 --num_iter=100  --iter_unit=batch --display_every=50 \
        --batch=256 --dali_cpu --log_dir=dali_log
}

pushd ../..
source ./qa/test_template.sh
popd
