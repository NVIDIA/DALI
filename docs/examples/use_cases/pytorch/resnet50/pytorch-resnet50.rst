ImageNet Training in PyTorch
============================

This implements training of popular model architectures, such as ResNet, AlexNet, and VGG on the ImageNet dataset.

This version has been modified to use DALI. It assumes that the dataset is raw JPEGs from the ImageNet dataset.
If offers CPU and GPU based pipeline for DALI - use dali_cpu switch to enable CPU one. For heavy GPU networks (like RN50) CPU based one is faster, for some lighter where CPU is the bottleneck like RN18 GPU is.
This version has been modified to use the DistributedDataParallel module in APEx instead of the one in upstream PyTorch. Please install APEx from `here <https://www.github.com/nvidia/apex>`_.

To run use the following commands

.. code-block:: bash

   ln -s /path/to/train/jpeg/ train
   ln -s /path/to/validation/jpeg/ val
   torchrun --nproc_per_node=NUM_GPUS main.py -a resnet50 --dali_cpu --b 128 \
            --loss-scale 128.0 --workers 4 --lr=0.4 --fp16-mode ./

Requirements
------------

.. role:: bash(code)
   :language: bash

- `APEx <https://www.github.com/nvidia/apex>`_ - optional (form PyTorch 1.6 it is part of the upstream so there is no need to install it separately), required for fp16 mode or distributed (multi-GPU) operation
- Install PyTorch from source, main branch of `PyTorch on github <https://www.github.com/pytorch/pytorch>`_
- :bash:`pip install -r requirements.txt`
- Download the ImageNet dataset and move validation images to labeled subfolders

  - To do this, you can use the following `script <https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh>`_

Training
--------

To train a model, run :fileref:`docs/examples/use_cases/pytorch/resnet50/main.py` with the desired model architecture and the path to the ImageNet dataset:

.. code-block:: bash

   python main.py -a resnet18 [imagenet-folder with train and val folders]

The default learning rate schedule starts at 0.1 and decays by a factor of 10 every 30 epochs. This is appropriate for ResNet and models with batch normalization, but too high for AlexNet and VGG. Use 0.01 as the initial learning rate for AlexNet or VGG:

.. code-block:: bash

   python main.py -a alexnet --lr 0.01 [imagenet-folder with train and val folders]

Data loaders
------------

- **dali**:
  Leverages a DALI pipeline along with DALI's PyTorch iterator for data loading, preprocessing, and augmentation.

- **dali_proxy**:
  Uses a DALI pipeline for preprocessing and augmentation while relying on PyTorch's data loader. DALI Proxy facilitates the transfer of data to DALI for processing.
  See :ref:`pytorch_dali_proxy`.

- **pytorch**:
  Employs the native PyTorch data loader for data preprocessing and augmentation.

Usage
-----

.. code-block:: bash
   main.py [-h] [--arch ARCH] [-j N] [--epochs N] [--start-epoch N] [-b N] [--lr LR] [--momentum M] [--weight-decay W] [--print-freq N] [--resume PATH]
                  [-e] [--pretrained] [--dali_cpu] [--data_loader {pytorch,dali,dali_proxy}] [--prof PROF] [--deterministic] [--fp16-mode]
                  [--loss-scale LOSS_SCALE] [--channels-last CHANNELS_LAST] [-t]
                  [DIR ...]

  PyTorch ImageNet Training

  positional arguments:
    DIR                   path(s) to dataset (if one path is provided, it is assumed to have subdirectories named "train" and "val"; alternatively, train and val paths can
                          be specified directly by providing both paths as arguments)

  options:
    -h, --help            show this help message and exit
    --arch ARCH, -a ARCH  model architecture: alexnet | convnext_base | convnext_large | convnext_small | convnext_tiny | densenet121 | densenet161 | densenet169 |
                          densenet201 | efficientnet_b0 | efficientnet_b1 | efficientnet_b2 | efficientnet_b3 | efficientnet_b4 | efficientnet_b5 | efficientnet_b6 |
                          efficientnet_b7 | efficientnet_v2_l | efficientnet_v2_m | efficientnet_v2_s | get_model | get_model_builder | get_model_weights | get_weight |
                          googlenet | inception_v3 | list_models | maxvit_t | mnasnet0_5 | mnasnet0_75 | mnasnet1_0 | mnasnet1_3 | mobilenet_v2 | mobilenet_v3_large |
                          mobilenet_v3_small | regnet_x_16gf | regnet_x_1_6gf | regnet_x_32gf | regnet_x_3_2gf | regnet_x_400mf | regnet_x_800mf | regnet_x_8gf |
                          regnet_y_128gf | regnet_y_16gf | regnet_y_1_6gf | regnet_y_32gf | regnet_y_3_2gf | regnet_y_400mf | regnet_y_800mf | regnet_y_8gf | resnet101 |
                          resnet152 | resnet18 | resnet34 | resnet50 | resnext101_32x8d | resnext101_64x4d | resnext50_32x4d | shufflenet_v2_x0_5 | shufflenet_v2_x1_0 |
                          shufflenet_v2_x1_5 | shufflenet_v2_x2_0 | squeezenet1_0 | squeezenet1_1 | swin_b | swin_s | swin_t | swin_v2_b | swin_v2_s | swin_v2_t | vgg11 |
                          vgg11_bn | vgg13 | vgg13_bn | vgg16 | vgg16_bn | vgg19 | vgg19_bn | vit_b_16 | vit_b_32 | vit_h_14 | vit_l_16 | vit_l_32 | wide_resnet101_2 |
                          wide_resnet50_2 (default: resnet18)
    -j N, --workers N     number of data loading workers (default: 4)
    --epochs N            number of total epochs to run
    --start-epoch N       manual epoch number (useful on restarts)
    -b N, --batch-size N  mini-batch size per process (default: 256)
    --lr LR, --learning-rate LR
                          Initial learning rate. Will be scaled by <global batch size>/256: args.lr = args.lr*float(args.batch_size*args.world_size)/256. A warmup schedule
                          will also be applied over the first 5 epochs.
    --momentum M          momentum
    --weight-decay W, --wd W
                          weight decay (default: 1e-4)
    --print-freq N, -p N  print frequency (default: 10)
    --resume PATH         path to latest checkpoint (default: none)
    -e, --evaluate        evaluate model on validation set
    --pretrained          use pre-trained model
    --dali_cpu            Runs CPU based version of DALI pipeline.
    --data_loader {pytorch,dali,dali_proxy}
                          Select data loader: "pytorch" for native PyTorch data loader, "dali" for DALI data loader, or "dali_proxy" for PyTorch dataloader with DALI proxy
                          preprocessing.
    --prof PROF           Only run 10 iterations for profiling.
    --deterministic       Enable deterministic behavior for reproducibility
    --fp16-mode           Enable half precision mode.
    --loss-scale LOSS_SCALE
                          Scaling factor for loss to prevent underflow in FP16 mode.
    --channels-last CHANNELS_LAST
                          Use channels last memory format for tensors.
    -t, --test            Launch test mode with preset arguments
