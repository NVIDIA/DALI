Single Shot MultiBox Detector training in PyTorch
============================

This example shows how DALI can be used in detection networks, specifically Single Shot Multibox Detector originally published by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, 
Scott Reed, Cheng-Yang Fu, Alexander C. Berg as `SSD: Single Shot MultiBox Detector <https://arxiv.org/abs/1512.02325>`_.

Code is based on `MLPerf example <https://github.com/mlperf/training/tree/master/single_stage_detector/ssd>`_ and has been modified to use DALI. 

To run training on 8 GPUs using half-precission with COCO 2017 dataset under ``/coco`` use following command:

.. code-block:: bash

   python -m torch.distributed.launch --nproc_per_node=8 ./main.py --warmup 300 --bs 64 --fp16 --data /coco/


Requirements
------------

- This example was tested with ``python3.5.2`` and it should work with later versions. It will not work with ``python2.7`` and earlier.

- Download `COCO 2017 dataset <http://cocodataset.org/#download>`_. You can also use:

  .. code-block:: bash

      dir=$(pwd)
      mkdir /coco; cd /coco
      curl -O http://images.cocodataset.org/zips/train2017.zip; unzip train2017.zip
      curl -O http://images.cocodataset.org/zips/val2017.zip; unzip val2017.zip
      curl -O http://images.cocodataset.org/annotations/annotations_trainval2017.zip; unzip annotations_trainval2017.zip
      cd $dir

- Install packages listed below into your ``python`` interpreter:

  ``numpy torch torchvision mlperf_compliance matplotlib Cython pycocotools``

Usage
-----

.. code-block:: bash

  usage: main.py [-h] --data DATA [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                [--eval-batch-size EVAL_BATCH_SIZE] [--seed SEED]
                [--evaluation [EVALUATION [EVALUATION ...]]]
                [--multistep [MULTISTEP [MULTISTEP ...]]]
                [--learning-rate LEARNING_RATE] [--momentum MOMENTUM]
                [--weight-decay WEIGHT_DECAY] [--warmup WARMUP]
                [--num-workers NUM_WORKERS] [--fp16] [--local_rank LOCAL_RANK]
                [--data_pipeline {dali,no_dali}]

All arguments with descriptions you can find in table below:

+---------------------------------------------+-----------------------------------------+
|                 Argument                    |              Description                |
+=============================================+=========================================+
| -h, --help                                  | show this help message and exit         |
+---------------------------------------------+-----------------------------------------+
| --data DATA, -d DATA                        | path to test and training data files    |
+---------------------------------------------+-----------------------------------------+
| --epochs EPOCHS, -e EPOCHS                  | number of epochs for training           |
+---------------------------------------------+-----------------------------------------+
| --batch-size BATCH_SIZE, -b BATCH_SIZE      | number of examples for each iteration   |
+---------------------------------------------+-----------------------------------------+
| --seed SEED, -s SEED                        | manually set random seed for torch      |
+---------------------------------------------+-----------------------------------------+
| --evaluation [EVALUATION [EVALUATION ...]]  | epochs at which to evaluate             |
+---------------------------------------------+-----------------------------------------+
| --multistep [MULTISTEP [MULTISTEP ...]]     | epochs at which to decay learning rate  |
+---------------------------------------------+-----------------------------------------+
| --learning-rate LEARNING_RATE               | learning rate                           |
+---------------------------------------------+-----------------------------------------+
| --momentum MOMENTUM                         | momentum argument for SGD optimizer     |
+---------------------------------------------+-----------------------------------------+
| --weight-decay WEIGHT_DECAY                 | weight decay value                      |
+---------------------------------------------+-----------------------------------------+
| --warmup WARMUP                             | number of warmup iterations             |
+---------------------------------------------+-----------------------------------------+
| --num-workers NUM_WORKERS                   | number of worker threads                |
+---------------------------------------------+-----------------------------------------+
| --fp16                                      | use half precission                     |
+---------------------------------------------+-----------------------------------------+
| --local_rank LOCAL_RANK                     | local rank of current process           |
+---------------------------------------------+-----------------------------------------+
| --data_pipeline {dali,no_dali}              | data pipeline to use for training       |
+---------------------------------------------+-----------------------------------------+
