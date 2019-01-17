#!/bin/bash -e

cd /opt/tensorflow/nvidia-examples/cnn/

mkdir -p idx-files/

NUM_GPUS=$(nvidia-smi -L | wc -l)

for file in $(ls /data/imagenet/train-val-tfrecord-480);
do
    echo ${file}
    python /opt/dali/tools/tfrecord2idx /data/imagenet/train-val-tfrecord-480/${file} \
        idx-files/${file}.idx;
done

function CLEAN_AND_EXIT {
    exit $1
}

LOG=dali.log
OUT=${LOG%.log}.dir

SECONDS=0
mpiexec --allow-run-as-root --bind-to socket -np ${NUM_GPUS} \
    python -u resnet.py --layers=18 \
    --data_dir=/data/imagenet/train-val-tfrecord-480-subset --data_idx_dir=idx-files/ \
    --precision=fp16   --num_iter=90  --iter_unit=epoch --display_every=50 \
    --batch=256 --use_dali=GPU --log_dir=$OUT \
    2>&1 | tee $LOG

RET=${PIPESTATUS[0]}
echo "Training ran in $SECONDS seconds"
if [[ $RET -ne 0 ]]; then
    echo "Error in training script."
    CLEAN_AND_EXIT 2
fi

MIN_TOP1=75.0
MIN_TOP5=92.0

TOP1=$(grep "^Top-1" $LOG | awk '{print $3}')
TOP5=$(grep "^Top-5" $LOG | awk '{print $3}')

if [[ -z "$TOP1" || -z "$TOP5" ]]; then
    echo "Incomplete output."
    CLEAN_AND_EXIT 3
fi

TOP1_RESULT=$(echo "$TOP1 $MIN_TOP1" | awk '{if ($1>=$2) {print "OK"} else { print "FAIL" }}')
TOP5_RESULT=$(echo "$TOP5 $MIN_TOP5" | awk '{if ($1>=$2) {print "OK"} else { print "FAIL" }}')

echo
printf "TOP-1 Accuracy: %.2f%% (expect at least %f%%) %s\n" $TOP1 $MIN_TOP1 $TOP1_RESULT
printf "TOP-5 Accuracy: %.2f%% (expect at least %f%%) %s\n" $TOP5 $MIN_TOP5 $TOP5_RESULT

if [[ "$TOP1_RESULT" == "OK" && "$TOP5_RESULT" == "OK" ]]; then
    CLEAN_AND_EXIT 0
fi
