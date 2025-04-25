.. _efficientnet_autoaugment:

EfficientNet for PyTorch with DALI and AutoAugment
==================================================

This example shows how DALI's implementation of automatic augmentations - most notably  `AutoAugment <https://arxiv.org/abs/1805.09501>`_ and `TrivialAugment <https://arxiv.org/abs/2103.10158>`_ - can be used in training. It shows the training of EfficientNet, an image classification model first described in  `EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks <https://arxiv.org/abs/1905.11946>`_.

The code is based on `NVIDIA Deep Learning Examples <https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Classification/ConvNets/efficientnet>`_ - it has been extended with DALI pipeline supporting automatic augmentations, which can be found in :fileref:`here <docs/examples/use_cases/pytorch/efficientnet/image_classification/dali.py>`.


Differences to the Deep Learning Examples configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

* The default values of the parameters were adjusted to values used in EfficientNet training.
* ``--data-backend`` parameter was changed to accept ``pytorch``, ``pytorch_optimized``, ``synthetic``, ``dali`` or ``dali_proxy``. It is set to ``dali`` by default.
* ``--dali-device`` was added to control placement of some of DALI operators.
* ``--augmentation`` was replaced with ``--automatic-augmentation``, now supporting ``disabled``, ``autoaugment``, and ``trivialaugment`` values.
* ``--workers`` defaults were halved to accommodate DALI. The value is automatically doubled when ``pytorch`` data loader is used. Thanks to this the default value performs well with both loaders.
* The model is restricted to EfficientNet-B0 architecture.


Data backends
^^^^^^^^^^^^^

This model uses the following data augmentation:

* For training:

  * Random resized crop to target images size (in this case 224)

    * Scale from 8% to 100%
    * Aspect ratio from 3/4 to 4/3

  * Random horizontal flip
  * [Optional: AutoAugment or TrivialAugment]
  * Normalization

* For inference:

  * Scale to target image size + additional size margin (in this case it is 224 + 32 = 266)
  * Center crop to target image size (in this case 224)
  * Normalization



Setup
^^^^^

The EfficientNet script operates on ImageNet 1k, a widely popular image classification dataset from the ILSVRC challenge.

1. Download the dataset from http://image-net.org/download-images

2. Extract the training data:

.. code-block:: bash

  mkdir train && mv ILSVRC2012_img_train.tar train/ && cd train
  tar -xvf ILSVRC2012_img_train.tar && rm -f ILSVRC2012_img_train.tar
  find . -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"
  tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
  cd ..

3. Extract the validation data and move the images to subfolders:

.. code-block:: bash

   mkdir val && mv ILSVRC2012_img_val.tar val/ && cd val && tar -xvf ILSVRC2012_img_val.tar
   wget -qO- https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh | bash

The directory in which the ``train/`` and ``val/`` directories are placed, is referred to as ``$PATH_TO_IMAGENET`` in this document.

4. Make sure you are either using the `NVIDIA PyTorch NGC container <https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch>`_ or you have `DALI <https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html>`_ and `PyTorch <https://pytorch.org/get-started/locally/>`_ installed.

5. Install `NVIDIA DLLogger <https://github.com/NVIDIA/dllogger>`_ and `pynvml <https://pypi.org/project/pynvml/>`_.


Running the model
^^^^^^^^^^^^^^^^^

Training
--------

To run training on a single GPU, use the ``main.py`` entry point:

* For FP32: ``python ./main.py --batch-size 64 $PATH_TO_IMAGENET``
* For AMP: ``python ./main.py --batch-size 64 --amp --static-loss-scale 128 $PATH_TO_IMAGENET``

You may need to adjust ``--batch-size`` parameter for your machine.

You can change the data loader and automatic augmentation scheme that are used by adding:

* ``--data-backend``: ``dali`` | ``dali_proxy`` | ``pytorch`` | ``synthetic``,
* ``--automatic-augmentation``: ``disabled`` | ``autoaugment`` | ``trivialaugment`` (the last one only for DALI),
* ``--dali-device``: ``cpu`` | ``gpu`` (only for DALI).

By default DALI GPU-variant with AutoAugment is used (``dali`` and ``dali_proxy`` backends).

Data Backends
-------------

- **dali**:
  Leverages a DALI pipeline along with DALI's PyTorch iterator for data loading, preprocessing, and augmentation.

- **dali_proxy**:
  Uses a DALI pipeline for preprocessing and augmentation while relying on PyTorch's data loader. DALI Proxy facilitates the transfer of data to DALI for processing.
  See :ref:`pytorch_dali_proxy`.

- **pytorch**: 
  Employs the native PyTorch data loader for data preprocessing and augmentation.

- **synthetic**: 
  Creates synthetic data on the fly, which is useful for testing and benchmarking purposes. This backend eliminates the need for actual datasets, providing a convenient way to simulate data loading.

For example to run the EfficientNet with AMP on a batch size of 128 with DALI using TrivialAugment you need to invoke:

.. code-block:: bash

  python ./main.py --amp --static-loss-scale 128 --batch-size 128 --data-backend dali --automatic-augmentation trivialaugment $PATH_TO_IMAGENET

To run on multiple GPUs, use the ``multiproc.py`` to launch the ``main.py`` entry point script, passing the number of GPUs as ``--nproc_per_node`` argument. For example, to run the model on 8 GPUs using AMP and DALI with AutoAugment you need to invoke:

.. code-block:: bash

  python ./multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128 --batch-size 128 --data-backend dali --automatic-augmentation autoaugment $PATH_TO_IMAGENET

To see the full list of available options and their descriptions, use the ``-h`` or ``--help`` command-line option, for example:

.. code-block:: bash

  python main.py -h


Training with standard configuration
------------------------------------

To run the training in a standard configuration (DGX A100/DGX-1V, AMP, 400 Epochs, DALI with AutoAugment) invoke the following command:

* for DGX1V-16G: ``python multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128 --batch-size 128 $PATH_TO_IMAGENET``

* for DGX-A100: ``python multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128 --batch-size 256 $PATH_TO_IMAGENET```

Benchmarking
------------

To run training benchmarks with different data loaders and automatic augmentations, you can use following commands, assuming that they are running on DGX1V-16G with 8 GPUs, 128 batch size and AMP:

.. code-block:: bash

  # Adjust the following variable to control where to store the results of the benchmark runs
  export RESULT_WORKSPACE=./

  # synthetic benchmark
  python multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128
                      --batch-size 128 --epochs 1 --prof 1000 --no-checkpoints
                      --training-only --data-backend synthetic
                      --workspace $RESULT_WORKSPACE
                      --report-file bench_report_synthetic.json $PATH_TO_IMAGENET

  # DALI without automatic augmentations
  python multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128
                      --batch-size 128 --epochs 4 --no-checkpoints --training-only
                      --data-backend dali --automatic-augmentation disabled
                      --workspace $RESULT_WORKSPACE
                      --report-file bench_report_dali.json $PATH_TO_IMAGENET

  # DALI with AutoAugment
  python multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128
                      --batch-size 128 --epochs 4 --no-checkpoints --training-only
                      --data-backend dali --automatic-augmentation autoaugment
                      --workspace $RESULT_WORKSPACE
                      --report-file bench_report_dali_aa.json $PATH_TO_IMAGENET

  # DALI with TrivialAugment
  python multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128
                      --batch-size 128 --epochs 4 --no-checkpoints --training-only
                      --data-backend dali --automatic-augmentation trivialaugment
                      --workspace $RESULT_WORKSPACE
                      --report-file bench_report_dali_ta.json $PATH_TO_IMAGENET

  # DALI proxy with AutoAugment
  python multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128
                      --batch-size 128 --epochs 4 --no-checkpoints --training-only
                      --data-backend dali_proxy --automatic-augmentation autoaugment
                      --workspace $RESULT_WORKSPACE
                      --report-file bench_report_dali_proxy_aa.json $PATH_TO_IMAGENET

  # DALI proxy with TrivialAugment
  python multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128
                      --batch-size 128 --epochs 4 --no-checkpoints --training-only
                      --data-backend dali_proxy --automatic-augmentation trivialaugment
                      --workspace $RESULT_WORKSPACE
                      --report-file bench_report_dali_proxy_ta.json $PATH_TO_IMAGENET

  # PyTorch without automatic augmentations
  python multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128
                      --batch-size 128 --epochs 4 --no-checkpoints --training-only
                      --data-backend pytorch --automatic-augmentation disabled
                      --workspace $RESULT_WORKSPACE
                      --report-file bench_report_pytorch.json $PATH_TO_IMAGENET

  # PyTorch with AutoAugment:
  python multiproc.py --nproc_per_node 8 ./main.py --amp --static-loss-scale 128
                      --batch-size 128 --epochs 4 --no-checkpoints --training-only
                      --data-backend pytorch --automatic-augmentation autoaugment
                      --workspace $RESULT_WORKSPACE
                      --report-file bench_report_pytorch_aa.json $PATH_TO_IMAGENET


Inference
---------

Validation is done every epoch, and can be also run separately on a checkpointed model.

.. code-block:: bash

  python ./main.py --evaluate --epochs 1 --resume <path to checkpoint>
                   -b <batch size> $PATH_TO_IMAGENET

To run inference on JPEG image, you have to first extract the model weights from checkpoint:

.. code-block:: bash

  python checkpoint2model.py --checkpoint-path <path to checkpoint>
                             --weight-path <path where weights will be stored>

Then, run the classification script:

.. code-block:: bash

  python classify.py --pretrained-from-file <path to weights from previous step>
                     --precision AMP|FP32 --image <path to JPEG image>

