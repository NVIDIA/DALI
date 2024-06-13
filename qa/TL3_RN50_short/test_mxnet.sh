#!/bin/bash -e

threshold=0.25
min_perf=4000

NUM_GPUS=`nvidia-smi -L | wc -l`

# Disable memory padding to avoid OOM errors
sed -i -e "s/'--dali-nvjpeg-memory-padding', int, 16/'--dali-nvjpeg-memory-padding', int, 0/g" /opt/mxnet/python/mxnet/io/dali_utils.py
# Make sure that nvImageCodec loads before MXNet to avoid getting libjpeg version mismatch
sed -i -e "s/import nvidia.dali.fn as fn/import nvidia.dali.fn as fn\nfrom nvidia import nvimgcodec/g" /opt/mxnet/python/mxnet/io/dali_utils.py

# TODO(janton): Remove the LD_PRELOAD with next nvImageCodec release
# Make sure that nvImageCodec's jpeg symbol load before MXNet to avoid getting libjpeg version mismatch
LD_PRELOAD=$(ls /usr/local/lib/python3.10/dist-packages/nvidia/nvimgcodec/extensions/libjpeg_turbo_ext.so*) \
python /opt/mxnet/example/image-classification/train_imagenet_runner \
       --data-root=/data/imagenet/train-val-recordio-passthrough/ -b 408 \
       -n $NUM_GPUS -e 5 --seed 42 --dali-threads 8 2>&1 | tee dali.log

cat dali.log  | grep -o "Validation-accuracy=0\.[0-9]*" | grep -o "0\.[0-9]*" > acc.log
cat dali.log  | grep -o "Speed: [0-9]*\.[0-9]*" | grep -o "[0-9]*\.[0-9]*" > speed.log

best=`awk 'BEGIN { max = -inf } { if ($1 > max) { max = $1 } } END { print max }' acc.log`
mean=`awk 'BEGIN { sum = 0; n = 0 } { sum += $1; n += 1 } END { print sum / n }' speed.log`

rm -rf acc.log speed.log

if [[ `echo "$best $threshold" | awk '{ print ($1 >= $2) ? "1" : "0" }'` -eq "0" ]]; then
    echo "acc = $best; TEST FAILED"
    exit -1
fi

if [[ `echo "$mean $min_perf" | awk '{ print ($1 >= $2) ? "1" : "0" }'` -eq "0" ]]; then
    echo "perf = $mean; TEST FAILED"
    exit -1
fi

echo "DONE! best accuracy = $best; mean speed = $mean samples/sec"
