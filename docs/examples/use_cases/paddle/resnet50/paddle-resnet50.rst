ResNet Training in PaddlePaddle
===============================

This is a demo showcasing ResNet50 training on ImageNet.
The code is based on `NVIDIA Deep Learning Examples <https://github.com/NVIDIA/DeepLearningExamples/tree/master/PaddlePaddle/Classification/RN50v1.5>`_

Data augmentation
-----------------

This model uses the following data augmentation:

- For training:

    - Normalization
    - Random resized crop to 224x224

        - Scale from 8% to 100%
        - Aspect ratio from 3/4 to 4/3

    - Random horizontal flip

- For inference:

    - Normalization
    - Scale to 256x256
    - Center crop to 224x224

Usage
-----

Install the necessary packages from requirements.txt before use.

The startup script is :fileref:`docs/examples/use_cases/paddle/resnet50/train.py`.

.. code-block:: bash

   # For single GPU training with AMP
  FLAGS_apply_pass_to_program=1 python -m paddle.distributed.launch \
    --gpus=0 train.py \
    --epochs 90 \
    --amp \
    --scale-loss 128.0 \
    --use-dynamic-loss-scaling \
    --data-layout NHWC

  # For 8 GPUs training with AMP
  FLAGS_apply_pass_to_program=1 python -m paddle.distributed.launch \
    --gpus=0,1,2,3,4,5,6,7 train.py \
    --epochs 90 \
    --amp \
    --scale-loss 128.0 \
    --use-dynamic-loss-scaling \
    --data-layout NHWC

  # For all available options
  python train.py --help
