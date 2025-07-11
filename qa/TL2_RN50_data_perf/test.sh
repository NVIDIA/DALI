#!/bin/bash -e
# used pip packages
pip_packages='opencv-python-headless nose'
target_dir=./dali/test/python

do_once() {
    NUM_GPUS=$(nvidia-smi -L | wc -l)
}

test_body() {
    if [ $(stat /data/imagenet/train-jpeg --format="%T" -f) != "ext2/ext3" ]; then
        echo "Not available locally, skipping the test"
        return 0
    fi

    python test_RN50_data_pipeline.py --gpus ${NUM_GPUS} -b 256 --workers 3 --prefetch 2 --decoder_type "legacy"
    python test_RN50_data_pipeline.py --gpus ${NUM_GPUS} -b 256 --workers 3 --prefetch 2 --decoder_type "experimental"
    python test_RN50_data_pipeline.py --gpus ${NUM_GPUS} -b 16 --workers 3 --prefetch 11 --decoder_type "legacy"
    python test_RN50_data_pipeline.py --gpus ${NUM_GPUS} -b 16 --workers 3 --prefetch 11 --decoder_type "experimental"
    # fp16 NHWC
    python test_RN50_data_pipeline.py --gpus ${NUM_GPUS} -b 256 --workers 3 --prefetch 2 --fp16 --nhwc --decoder_type "legacy"
    python test_RN50_data_pipeline.py --gpus ${NUM_GPUS} -b 256 --workers 3 --prefetch 2 --fp16 --nhwc --decoder_type "experimental"
    python test_RN50_data_pipeline.py --gpus ${NUM_GPUS} -b 16 --workers 3 --prefetch 11 --fp16 --nhwc --decoder_type "legacy"
    python test_RN50_data_pipeline.py --gpus ${NUM_GPUS} -b 16 --workers 3 --prefetch 11 --fp16 --nhwc --decoder_type "experimental"
    # Paralell ExternalSource:
    python ./test_RN50_external_source_parallel_data.py /data/imagenet/train-jpeg/ --workers 6 --py_workers 6 --epochs 3 --batch_size 256 --reader_queue_depth 2 --worker_init fork --benchmark_iters 500 --gpus ${NUM_GPUS} --test_pipes parallel
    python ./test_RN50_external_source_parallel_data.py /data/imagenet/train-jpeg/ --workers 6 --py_workers 6 --epochs 3 --batch_size 256 --reader_queue_depth 2 --worker_init spawn --benchmark_iters 500 --gpus ${NUM_GPUS} --test_pipes parallel
    python ./test_RN50_external_source_parallel_data.py /data/imagenet/train-jpeg/ --workers 6 --py_workers 6 --epochs 3 --batch_size 256 --reader_queue_depth 2  --benchmark_iters 500 --gpus ${NUM_GPUS} --test_pipes file_reader
    python ./test_RN50_external_source_parallel_data.py /data/imagenet/train-jpeg/ --workers 6 --py_workers 6 --epochs 3 --batch_size 256 --reader_queue_depth 2 --benchmark_iters 250 --gpus ${NUM_GPUS} --test_pipes scalar
}

pushd ../..
source ./qa/test_template.sh
popd
