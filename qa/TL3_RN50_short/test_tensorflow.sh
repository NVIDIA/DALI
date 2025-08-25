#!/bin/bash -e

# enable compat for CUDA 13 if the test image doesn't support it yet
source <(echo "set -x"; cat ../setup_test_common.sh; echo "set +x")

install_cuda_compat

cd /opt/dali/docs/examples/use_cases/tensorflow/resnet-n

mkdir -p idx-files/

NUM_GPUS=$(nvidia-smi -L | wc -l)

DATA_SET_DIR=/data/imagenet/train-val-tfrecord
for file in $(ls $DATA_SET_DIR/*-of-*);
do
    file=$(basename ${file})
    echo ${file}
    python /opt/dali/tools/tfrecord2idx $DATA_SET_DIR/${file} \
        idx-files/${file}.idx &
done
wait

function CLEAN_AND_EXIT {
    exit $1
}

LOG=dali.log
OUT=${LOG%.log}.dir
mkdir -p $OUT

SECONDS=0

# turn off SHARP to avoid NCCL errors
export NCCL_NVLS_ENABLE=0

export TF_XLA_FLAGS="--tf_xla_enable_lazy_compilation=false"

mpiexec --allow-run-as-root --bind-to none -np ${NUM_GPUS} \
    python -u resnet.py \
    --data_dir=$DATA_SET_DIR --data_idx_dir=idx-files/ \
    --precision=fp16 --num_iter=5 --iter_unit=epoch --display_every=50 \
    --batch=256 --use_xla --log_dir=$OUT --dali_threads 8 \
    --dali_mode="GPU" 2>&1 | tee $LOG

RET=${PIPESTATUS[0]}
echo "Training ran in $SECONDS seconds"
if [[ $RET -ne 0 ]]; then
    echo "Error in training script."
    CLEAN_AND_EXIT 2
fi


MIN_TOP1=0.25
MIN_TOP5=0.50
MIN_PERF=4000

TOP1=$(grep "loss:" $LOG | awk '{print $18}' | tail -1)
TOP5=$(grep "loss:" $LOG | awk '{print $21}' | tail -1)

PERF=$(cat "$LOG" | grep "^global_step:" | awk " { sum += \$4; count+=1 } END {print sum/count}")

if [[ -z "$TOP1" || -z "$TOP5" ]]; then
    echo "Incomplete output."
    CLEAN_AND_EXIT 3
fi

TOP1_RESULT=$(echo "$TOP1 $MIN_TOP1" | awk '{if ($1>=$2) {print "OK"} else { print "FAIL" }}')
TOP5_RESULT=$(echo "$TOP5 $MIN_TOP5" | awk '{if ($1>=$2) {print "OK"} else { print "FAIL" }}')
PERF_RESULT=$(echo "$PERF $MIN_PERF" | awk '{if ($1>=$2) {print "OK"} else { print "FAIL" }}')

echo
printf "TOP-1 Accuracy: %.2f%% (expect at least %f%%) %s\n" $TOP1 $MIN_TOP1 $TOP1_RESULT
printf "TOP-5 Accuracy: %.2f%% (expect at least %f%%) %s\n" $TOP5 $MIN_TOP5 $TOP5_RESULT
printf "mean speed %.2f (expect at least %f) samples/sec %s\n" $PERF $MIN_PERF $PERF_RESULT

# check perf only if data is locally available
if [ $(stat /data/imagenet/train-val-tfrecord --format="%T" -f) == "ext2/ext3" ] && [ "$PERF_RESULT" != "OK" ]; then
    CAN_AND_EXIT 4
fi

if [[ "$TOP1_RESULT" == "OK" && "$TOP5_RESULT" == "OK" ]]; then
    CLEAN_AND_EXIT 0
fi

CLEAN_AND_EXIT 4
