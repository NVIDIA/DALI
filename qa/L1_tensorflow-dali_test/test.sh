#!/bin/bash -e
pip_packages=""

pushd ../..

cd docs/examples/tensorflow/demo

mkdir -p idx-files/

NUM_GPUS=$(nvidia-smi -L | wc -l)

CUDA_VERSION=$(nvcc --version | grep -E ".*release ([0-9]+)\.([0-9]+).*" | sed 's/.*release \([0-9]\+\)\.\([0-9]\+\).*/\1\2/')
# from 1.13.1 CUDA 10 is supported but not CUDA 9
# MPI is present in CUDA 10 image already so no need to build it
if [ "${CUDA_VERSION}" == "100" ]; then
    pip install tensorflow-gpu==1.13.1
else
    pip install tensorflow-gpu==1.12

    OPENMPI_VERSION=3.0.0
    wget -q -O - https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-${OPENMPI_VERSION}.tar.gz | tar -xzf -
    cd openmpi-${OPENMPI_VERSION}
    ./configure --enable-orterun-prefix-by-default --prefix=/usr/local/mpi --disable-getpwuid -with-cma=no
    make -j"$(nproc)" install
    cd .. && rm -rf openmpi-${OPENMPI_VERSION}
    echo "/usr/local/mpi/lib" >> /etc/ld.so.conf.d/openmpi.conf && ldconfig
    export PATH=$PATH:/usr/local/mpi/bin

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
    python ../../../../tools/tfrecord2idx /data/imagenet/train-val-tfrecord-480-subset/${file} \
        idx-files/${file}.idx;
done

# OMPI_MCA_blt is needed if running in a container w/o cap SYS_PRACE
# More info: https://github.com/radiasoft/devops/issues/132
export OMPI_MCA_blt=self,sm,tcp

test_body() {
    # test code
    mpiexec --allow-run-as-root --bind-to socket -np ${NUM_GPUS} \
        python -u resnet.py --layers=18 \
        --data_dir=/data/imagenet/train-val-tfrecord-480-subset --data_idx_dir=idx-files/ \
        --precision=fp16 --num_iter=100  --iter_unit=batch --display_every=50 \
        --batch=256 --dali_cpu --log_dir=dali_log
}

source ../../../../qa/test_template.sh

popd
