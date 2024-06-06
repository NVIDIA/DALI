You Only Look Once v4 with TensorFlow and DALI
==============================================

This example presents a sample implementation of a YOLOv4 network,
based on the following paper – `Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao,
YOLOv4: Optimal Speed and Accuracy of Object Detection <https://arxiv.org/pdf/2004.10934.pdf>`_.

The implementation is based on the code available in the `darknet repository <https://github.com/AlexeyAB/darknet>`_.

To run training from scratch on all available GPUs with DALI, run the following command:

.. code-block:: bash

  python src/main.py train /coco/train2017 /coco/annotations/instances_train2017.json \
    -b 8 -e 6 -s 1000 -o output.h5 \
    --pipeline dali-gpu --multigpu --use_mosaic

To save checkpoints each epoch, add the following flag to the command:

.. code-block:: bash

  --ckpt_dir /ckpt

Resume the training by providing an appropriate path to the ``-w`` flag:

.. code-block:: bash

  -w /ckpt/epoch_3.h5

To perform evaluation every second epoch, add the following flags:

.. code-block:: bash

  --eval_file_root /coco/val2017 \
  --eval_annotations /coco/annotations/instances_val2017.json \
  --eval_frequency 2 --eval_steps 500

To evaluate trained model, run the following command:

.. code-block:: bash

  python src/main.py eval /coco/val2017 /coco/annotations/instances_val2017.json \
    -w output.h5 -b 1 -s 5000

To perform an inference and display the results on screen, run the following command:

.. code-block:: bash

  python src/main.py infer image.png -w output.h5 -c coco-labels.txt

Requirements
------------

- This example was tested using ``python 3.8`` and ``tensorflow 2.4.1`` and should work on later versions.

- The model requires `COCO 2017 dataset <http://cocodataset.org/#download>`_ for training.

- Besides ``TensorFlow``, the following python packages are also required:

  ``matplotlib tensorflow-addons pycocotools``

Usage
-----

Training
^^^^^^^^

.. code-block:: bash

  usage: main.py train [-h] file_root annotations
    [--batch_size BATCH_SIZE] [--epochs EPOCHS] [--steps STEPS] [--output OUTPUT]
    [--start_weights START_WEIGHTS] [--log_dir LOG_DIR] [--ckpt_dir CKPT_DIR]
    [--pipeline PIPELINE] [--multigpu] [--use_mosaic] [--learning_rate LEARNING_RATE]
    [--eval_file_root EVAL_FILE_ROOT] [--eval_annotations EVAL_ANNOTATIONS]
    [--eval_steps EVAL_STEPS] [--eval_frequency EVAL_FREQUENCY]
    [--seed SEED]

+-------------------------------------------------+------------------------------------------------------------------+
|                    Argument                     |                            Description                           |
+=================================================+==================================================================+
| -h, --help                                      | show this help message and exit                                  |
+-------------------------------------------------+------------------------------------------------------------------+
| file_root                                       | path to folder containing train coco2017 images                  |
+-------------------------------------------------+------------------------------------------------------------------+
| annotations                                     | path to instances_train2017.json file                            |
+-------------------------------------------------+------------------------------------------------------------------+
| --batch_size BATCH_SIZE, -b BATCH_SIZE          | number of images per training step, default = 8                  |
+-------------------------------------------------+------------------------------------------------------------------+
| --epochs EPOCHS, -e EPOCHS                      | number of training epochs, default = 5                           |
+-------------------------------------------------+------------------------------------------------------------------+
| --steps STEPS, -s STEPS                         | number of training steps per epoch, default = 1000               |
+-------------------------------------------------+------------------------------------------------------------------+
| --output OUTPUT, -o OUTPUT                      | path to a .h5 output file for trained model, default = output.h5 |
+-------------------------------------------------+------------------------------------------------------------------+
| --start_weights START_WEIGHTS, -w START_WEIGHTS | initial weights file in h5 or YOLO format                        |
+-------------------------------------------------+------------------------------------------------------------------+
| --log_dir LOG_DIR                               | path to a directory for TensorBoard logs                         |
+-------------------------------------------------+------------------------------------------------------------------+
| --ckpt_dir CKPT_DIR                             | path to a directory for checkpoint files                         |
+-------------------------------------------------+------------------------------------------------------------------+
| --pipeline PIPELINE                             | either dali_gpu, dali_cpu or numpy                               |
+-------------------------------------------------+------------------------------------------------------------------+
| --multigpu                                      | if present, training is run using all available GPUs             |
+-------------------------------------------------+------------------------------------------------------------------+
| --use_mosaic                                    | if present, mosaic data augmentation is used                     |
+-------------------------------------------------+------------------------------------------------------------------+
| --learning_rate LEARNING_RATE                   | learning rate for training, default = 1e-3                       |
+-------------------------------------------------+------------------------------------------------------------------+
| --eval_file_root EVAL_FILE_ROOT                 | path to folder containing val coco2017 images                    |
+-------------------------------------------------+------------------------------------------------------------------+
| --eval_annotations EVAL_ANNOTATIONS             | path to instances_val2017.json file                              |
+-------------------------------------------------+------------------------------------------------------------------+
| --eval_steps EVAL_STEPS                         | number of images per evaluation step, default = 5000             |
+-------------------------------------------------+------------------------------------------------------------------+
| --eval_frequency EVAL_FREQUENCY                 | number of training epochs between each evaluation, default = 5   |
+-------------------------------------------------+------------------------------------------------------------------+
| --seed SEED                                     | seed for DALI and TensorFlow                                     |
+-------------------------------------------------+------------------------------------------------------------------+


Inference
^^^^^^^^^

.. code-block:: bash

  usage: main.py infer [-h] image [--weights WEIGHTS] [--classes CLASSES]
                       [--output OUTPUT]

+-------------------------------------------------+-----------------------------------------------------+
|                    Argument                     |                    Description                      |
+=================================================+=====================================================+
| -h, --help                                      | show this help message and exit                     |
+-------------------------------------------------+-----------------------------------------------------+
| image                                           | path to an image to perform inference on            |
+-------------------------------------------------+-----------------------------------------------------+
| --weights WEIGHTS, -w WEIGHTS                   | path to a trained weights file in h5 or YOLO format |
+-------------------------------------------------+-----------------------------------------------------+
| --classes CLASSES, -c CLASSES                   | path to a coco-labels.txt file                      |
+-------------------------------------------------+-----------------------------------------------------+
| --output OUTPUT, -o OUTPUT                      | path to an output image                             |
+-------------------------------------------------+-----------------------------------------------------+


Evaluation
^^^^^^^^^^

.. code-block:: bash

  usage: main.py eval [-h] file_root annotations [--weights WEIGHTS]
                      [--batch_size BATCH_SIZE] [--steps STEPS]

+-------------------------------------------------+-----------------------------------------------------+
|                    Argument                     |                    Description                      |
+=================================================+=====================================================+
| -h, --help                                      | show this help message and exit                     |
+-------------------------------------------------+-----------------------------------------------------+
| file_root                                       | path to folder containing val coco2017 images       |
+-------------------------------------------------+-----------------------------------------------------+
| annotations                                     | path to instances_val2017.json file                 |
+-------------------------------------------------+-----------------------------------------------------+
| --weights WEIGHTS, -w WEIGHTS                   | path to a trained weights file in h5 or YOLO format |
+-------------------------------------------------+-----------------------------------------------------+
| --batch_size BATCH_SIZE, -b BATCH_SIZE          | number of images per evaluation step, default = 1   |
+-------------------------------------------------+-----------------------------------------------------+
| --steps STEPS, -s STEPS                         | number of evaluation steps, default = 1000          |
+-------------------------------------------------+-----------------------------------------------------+
