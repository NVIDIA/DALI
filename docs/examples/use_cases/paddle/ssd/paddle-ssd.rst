Single Shot MultiBox Detector Training in PaddlePaddle
======================================================

This demo shows how to use DALI with PaddlePaddle for training Single Shot Multibox Detector (`SSD: Single Shot MultiBox Detector <https://arxiv.org/abs/1512.02325>`_).

The model is designed to train on 8 GPUs with a mini-batch size of 8 per GPU, to train on all GPUs of the system, simply run:

.. code-block:: bash

   python train.py -b 8 [path to coco dataset]


Requirements
------------

- Download and extract the `COCO2017 dataset <http://cocodataset.org/#download>`_.

  .. code-block:: bash

      wget http://images.cocodataset.org/zips/train2017.zip
      wget http://images.cocodataset.org/zips/val2017.zip
      wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
      unzip train2017.zip
      unzip val2017.zip
      unzip annotations_trainval2017.zip

- Install the following python packages via pip or other means.

  - `PaddlePaddle <https://www.paddlepaddle.org>`_ (1.6 or above)

  - `Nvidia DALI <https://github.com/NVIDIA/DALI>`_

Usage
-----

.. code-block:: bash

   usage: train.py [-h] [-j N] [-b N] [--lr LR] [--momentum M] [--weight-decay W]
                   [--print-freq N] [--ckpt-freq N]
                   DIR

   Paddle Single Shot MultiBox Detector Training

   positional arguments:
     DIR                   path to dataset

   optional arguments:
     -h, --help            show this help message and exit
     -j N, --num_threads N
                           number of threads (default: 4)
     -b N, --batch-size N  mini-batch size (default: 8)
     --lr LR, --learning-rate LR
                           initial learning rate
     --momentum M          momentum
     --weight-decay W, --wd W
                           weight decay (default: 1e-4)
     --print-freq N, -p N  print frequency (default: 10)
     --ckpt-freq N, -c N   checkpoint frequency (default: 5000)
