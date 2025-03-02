#!/bin/bash -e
# used pip packages
pip_packages='pillow torchvision torch opencv-python nose'
target_dir=./dali/test/python
one_config_only=true

do_once() {
    NUM_GPUS=$(nvidia-smi -L | wc -l)
}

test_body() {
    for fw in "pytorch"; do
        python test_RN50_data_fw_iterators.py --framework ${fw} --gpus ${NUM_GPUS} -b 13 \
            --workers 3 --prefetch 2 --epochs 3
    done
    start=`date +%s`
    (sleep 10 && pkill -HUP ls && true) &
    (ls /data/imagenet/train-jpeg > /dev/null && pkill -HUP sleep) &
    wait
    end=`date +%s`
    runtime=$((end-start))
    echo "Data access time: $runtime seconds"
    if [ $runtime -gt 3 ]; then
        echo "Data access time is greater than 3 seconds, skipping the test"
        return 0
    fi
    torchrun --nproc_per_node=${NUM_GPUS} ./test_RN50_external_source_parallel_train_ddp.py /data/imagenet/train-jpeg/ --workers 6 --py_workers 6 --epochs 3 --batch_size 256 --reader_queue_depth 2 --worker_init fork --benchmark_iters 500 --test_pipes parallel
    torchrun --nproc_per_node=${NUM_GPUS} ./test_RN50_external_source_parallel_train_ddp.py /data/imagenet/train-jpeg/ --workers 6 --py_workers 6 --epochs 3 --batch_size 256 --reader_queue_depth 2 --worker_init spawn --benchmark_iters 500 --test_pipes parallel
    torchrun --nproc_per_node=${NUM_GPUS} ./test_RN50_external_source_parallel_train_ddp.py /data/imagenet/train-jpeg/ --workers 6 --py_workers 6 --epochs 3 --batch_size 256 --reader_queue_depth 2 --benchmark_iters 500 --test_pipes file_reader
    torchrun --nproc_per_node=${NUM_GPUS} ./test_RN50_external_source_parallel_train_ddp.py /data/imagenet/train-jpeg/ --workers 6 --py_workers 6 --epochs 3 --batch_size 256 --reader_queue_depth 2 --benchmark_iters 250 --test_pipes scalar
}

pushd ../..
source ./qa/test_template.sh
popd
