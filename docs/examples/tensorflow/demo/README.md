# ResNet-N with TensorFlow and DALI

This demo implements residual networks model and use DALI for the data augmentation pipeline 
from [the original paper](https://arxiv.org/pdf/1512.03385.pdf).

Common utilities for defining the network and performing basic training are
located in the nvutils directory. Use of nvutils is demonstrated in the model
scripts.

For parallelization, we use the Horovod distribution framework, which works in
concert with MPI. To train ResNet-50 (--layers=50) using 8 V100 GPUs, for example on DGX-1,
use the following command (--dali\_cpu indicates to the script to use HostDecoder instead of nvJPEGDecoder):

```
$ mpiexec --allow-run-as-root --bind-to socket -np 8 python resnet.py \
                                                     --layers=50 \
                                                     --data_dir=/data/imagenet \
                                                     --precision=fp16 \
                                                     --log_dir=/output/resnet50 \
                                                     --dali_cpu
```


Here we have assumed that imagenet is stored in tfrecord format in the directory
'/data/imagenet'. After training completes, evaluation is performed using the
validation dataset.

Some common training parameters can tweaked from the command line. Others must
be configured within the network scripts themselves.

Original scripts modified from `nvidia-examples` scripts in
[NGC TensorFlow Container](https://www.nvidia.com/en-us/gpu-cloud/deep-learning-containers/)
