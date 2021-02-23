ResNet Training in PaddlePaddle
===============================

This simple demo showcases ResNet50 training on ImageNet.

Run it with the commands below:

.. code-block:: bash

   python -m paddle.distributed.launch --selected_gpus 0,1,2,3,4,5,6,7 train.py -b 128 -j 4 [imagenet-folder with train and val folders]

Training
--------

To train the model, run :fileref:`docs/examples/use_cases/paddle/resnet50/main.py` with the desired ResNet depth and the path to the ImageNet dataset:

.. code-block:: bash

   python -m paddle.distributed.launch --selected_gpus 0,1,2,3,4,5,6,7 main.py -d 50 [imagenet-folder with train and val folders]

The training schedule in `He et al. 2015 <https://arxiv.org/abs/1512.03385>`_ was used where learning rate starts at 0.1 and decays by a factor of 10 every 30 epochs.

Usage
-----

.. code-block:: bash

   usage: main.py [-h] [-d N] [-j N] [-b N] [--lr LR] [--momentum M]
                   [--weight-decay W] [--print-freq N]
                   DIR

   Paddle ImageNet Training

   positional arguments:
     DIR                   path to dataset (should have subdirectories named
                           "train" and "val"

   optional arguments:
     -h, --help            show this help message and exit
     -d N, --depth N       number of layers (default: 50)
     -j N, --num_threads N
                           number of threads (default: 4)
     -b N, --batch-size N  mini-batch size (default: 256)
     --lr LR, --learning-rate LR
                           initial learning rate
     --momentum M          momentum
     --weight-decay W, --wd W
                           weight decay (default: 1e-4)
     --print-freq N, -p N  print frequency (default: 10)
