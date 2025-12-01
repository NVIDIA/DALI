ResNet-N with TensorFlow and DALI
=================================

This demo implements residual networks model and use DALI for the data
augmentation pipeline from `the original paper`_.

It implements the ResNet50 v1.5 CNN model and demonstrates efficient
single-node training on multi-GPU systems. They can be used for benchmarking, or
as a starting point for implementing and training your own network.

Common utilities for defining CNN networks and performing basic training are
located in the nvutils directory inside :fileref:`docs/examples/use_cases/tensorflow/resnet-n`.
The utilities are written in Tensorflow 2.0.
Use of nvutils is demonstrated in the model script (i.e. resnet.py). The scripts
support both Keras Fit/Compile and Custom Training Loop (CTL) modes with
Horovod.

To use DALI pipeline for data loading and preprocessing ``--dali_mode=GPU`` or
``--dali_mode=CPU``

Training in Keras Fit/Compile mode
----------------------------------
For the full training on 8 GPUs::

    mpiexec --allow-run-as-root --bind-to socket -np 8 \
      python resnet.py --num_iter=90 --iter_unit=epoch \
      --data_dir=/data/imagenet/train-val-tfrecord/ \
      --precision=fp16 --display_every=100 \
      --export_dir=/tmp --dali_mode="GPU"

For the benchmark training on 8 GPUs::

    mpiexec --allow-run-as-root --bind-to socket -np 8 \
      python resnet.py --num_iter=400 --iter_unit=batch \
      --data_dir=/data/imagenet/train-val-tfrecord/ \
      --precision=fp16 --display_every=100 --dali_mode="GPU"


Predicting in Keras Fit/Compile mode
------------------------------------
For predicting with previously saved mode in `/tmp`::

    python resnet.py --predict --export_dir=/tmp --dali_mode="GPU"


Training in CTL (Custom Training Loop) mode
-------------------------------------------
For the full training on 8 GPUs::

    mpiexec --allow-run-as-root --bind-to socket -np 8 \
      python resnet_ctl.py --num_iter=90 --iter_unit=epoch \
      --data_dir=/data/imagenet/train-val-tfrecord/ \
      --precision=fp16 --display_every=100 \
      --export_dir=/tmp --dali_mode="GPU"

For the benchmark training on 8 GPUs::

    mpiexec --allow-run-as-root --bind-to socket -np 8 \
      python resnet_ctl.py --num_iter=400 --iter_unit=batch \
      --data_dir=/data/imagenet/train-val-tfrecord/ \
      --precision=fp16 --display_every=100 --dali_mode="GPU"

Predicting in CTL (Custom Training Loop) mode
---------------------------------------------
For predicting with previously saved mode in `/tmp`::

    python resnet_ctl.py --predict --export_dir=/tmp --dali_mode="GPU"

Other useful options
--------------------
To use tensorboard (Note, `/tmp/some_dir` needs to be created by users)::

    --tensorboard_dir=/tmp/some_dir


To export saved model at the end of training (Note, `/tmp/some_dir` needs to be created by users)::

    --export_dir=/tmp/some_dir

To store checkpoints at the end of every epoch (Note, `/tmp/some_dir` needs to be created by users)::

    --log_dir=/tmp/some_dir

To enable XLA::

    --use_xla


Requirements
~~~~~~~~~~~~

TensorFlow
^^^^^^^^^^

::

   pip install tensorflow-gpu==2.4.1

OpenMPI
^^^^^^^

::

   wget -q -O - \
     https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz \
     | tar -xz
   cd openmpi-3.0.0
   ./configure --enable-orterun-prefix-by-default --with-cuda \
               --prefix=/usr/local/mpi --disable-getpwuid
   make -j"$(nproc)" install
   cd .. && rm -rf openmpi-3.0.0
   echo "/usr/local/mpi/lib" >> /etc/ld.so.conf.d/openmpi.conf && ldconfig
   export PATH=/usr/local/mpi/bin:$PATH

The following works around a segfault in OpenMPI 3.0 when run within a
single node without ssh being installed.

::

   /bin/echo -e '#!/bin/bash'\
   '\ncat <<EOF'\
   '\n======================================================================'\
   '\nTo run a multi-node job, install an ssh client and clear plm_rsh_agent'\
   '\nin '/usr/local/mpi/etc/openmpi-mca-params.conf'.'\
   '\n======================================================================'\
   '\nEOF'\
   '\nexit 1' >> /usr/local/mpi/bin/rsh_warn.sh && \
       chmod +x /usr/local/mpi/bin/rsh_warn.sh && \
       echo "plm_rsh_agent = /usr/local/mpi/bin/rsh_warn.sh" \
       >> /usr/local/mpi/etc/openmpi-mca-params.conf

Horovod
^^^^^^^

::

   export HOROVOD_GPU_ALLREDUCE=NCCL
   export HOROVOD_NCCL_INCLUDE=/usr/include
   export HOROVOD_NCCL_LIB=/usr/lib/x86_64-linux-gnu
   export HOROVOD_NCCL_LINK=SHARED
   export HOROVOD_WITHOUT_PYTORCH=1
   pip install horovod==0.21.0

.. _the original paper: https://arxiv.org/pdf/1512.03385.pdf
.. _NGC TensorFlow Container: https://www.nvidia.com/en-us/gpu-cloud/deep-learning-containers/
