#!/bin/bash -e
apt_packages="openmpi-bin libopenmpi-dev"
pip_packages="opencv-python tensorflow-gpu mpi4py horovod"

pushd ../..

cd docs/examples/tensorflow/demo

mkdir -p idx-files/

NUM_GPUS=$(nvidia-smi -L | wc -l)

test_body() {
    # test code
    for file in $(ls /data/imagenet/train-val-tfrecord-480-subset);
    do
        python ../../../../tools/tfrecord2idx /data/imagenet/train-val-tfrecord-480-subset/${file} \
            idx-files/${file}.idx;
    done

    mpiexec --allow-run-as-root --bind-to socket -np ${NUM_GPUS} \
        python -u resnet.py --layers=18 \
        --data_dir=/data/imagenet/train-val-tfrecord-480-subset --data_idx_dir=idx-files/ \
        --precision=fp16 --num_iter=100  --iter_unit=batch --display_every=50 \
        --batch=256 --dali_cpu --log_dir=dali-log
}

source ../../../../qa/test_template.sh

popd
