#!/bin/bash -e

threshold=0.75
min_perf=10000

NUM_GPUS=`nvidia-smi -L | wc -l`

python /opt/mxnet/example/image-classification/train_imagenet_runner \
       --data-root=/data/imagenet/train-val-recordio-passthrough/ -b 208 \
       -n $NUM_GPUS --seed 42 2>&1 | tee dali.log

cat dali.log  | grep -o "Validation-accuracy=0\.[0-9]*" > tmp2.log
cat dali.log  | grep -o "Speed: [0-9]*\.[0-9]*" > tmp3.log
cat tmp2.log | grep -o "0\.[0-9]*" > dali.log
cat tmp3.log | grep -o "[0-9]*\.[0-9]*" > tmp2.log

best=`awk 'BEGIN { max = -inf } { if ($1 > max) { max = $1 } } END { print max }' dali.log`
mean=`awk 'BEGIN { sum = 0; n = 0 } { sum += $1; n += 1 } END { print sum / n }' tmp2.log`

rm tmp2.log tmp3.log

if [[ `echo "$best $threshold" | awk '{ print ($1 >= $2) ? "1" : "0" }'` -eq "0" ]]; then
    echo "acc = $best; TEST FAILED"
    exit -1
fi

if [[ `echo "$mean $min_perf" | awk '{ print ($1 >= $2) ? "1" : "0" }'` -eq "0" ]]; then
    echo "perf = $mean; TEST FAILED"
    exit -1
fi

echo "DONE! best accuracy = $best; mean speed = $mean samples/sec"
