Single Shot MultiBox Detector training in PyTorch
============================

This example shows how DALI can be used in detection networks, specifically Single Shot Multibox Detector originally published by Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy, 
Scott Reed, Cheng-Yang Fu, Alexander C. Berg as `SSD: Single Shot MultiBox Detector <https://arxiv.org/abs/1512.02325>`_.

Code is based on `MLPerf example <https://github.com/mlperf/training/tree/master/single_stage_detector/ssd>`_ and has been modified to use DALI. 

To run use following command:

.. code-block:: bash

   python train.py


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

  python train.py [-h] [--data DATA] [--epochs EPOCHS] [--batch-size BATCH_SIZE]
                  [--seed SEED] [--threshold THRESHOLD] [--iteration ITERATION]
                  [--checkpoint CHECKPOINT] [--no-save]
                  [--evaluation [EVALUATION [EVALUATION ...]]]

For example, if you have COCO data in ``/data/coco2017`` and wish to train for 80 epochs you could use:

  .. code-block:: bash

    python train.py --data=/data/coco2017 --epochs=80

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
| --threshold THRESHOLD, -t THRESHOLD         | stop training early at threshold        |
+---------------------------------------------+-----------------------------------------+
| --iteration ITERATION                       | iteration to start from                 |
+---------------------------------------------+-----------------------------------------+
| --checkpoint CHECKPOINT                     | path to model checkpoint file           |
+---------------------------------------------+-----------------------------------------+
| --no-save                                   | save model checkpoints                  |
+---------------------------------------------+-----------------------------------------+
| --evaluation [EVALUATION [EVALUATION ...]]  | iterations at which to evaluate         |
+---------------------------------------------+-----------------------------------------+
