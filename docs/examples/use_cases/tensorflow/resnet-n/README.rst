ResNet-N with TensorFlow and DALI
=================================

This demo implements residual networks model and use DALI for the data
augmentation pipeline from `the original paper`_.

Common utilities for defining the network and performing basic training
are located in the nvutils directory. Use of nvutils is demonstrated in
the model scripts available in :fileref:`docs/examples/tensorflow/demo/resnet.py`.

For parallelization, we use the Horovod distribution framework, which
works in concert with MPI. To train ResNet-50 (``--layers=50``) using 8
V100 GPUs, for example on DGX-1, use the following command
(``--dali_cpu`` indicates to the script to use CPU backend for DALI):

::

   $ mpiexec --allow-run-as-root --bind-to socket -np 8 python resnet.py \
                                                        --layers=50 \
                                                        --data_dir=/data/imagenet \
                                                        --data_idx_dir=/data/imagenet-idx \
                                                        --precision=fp16 \
                                                        --log_dir=/output/resnet50 \
                                                        --dali_cpu

Here we have assumed that imagenet is stored in tfrecord format in the
directory '/data/imagenet'. After training completes, evaluation is
performed using the validation dataset.

Some common training parameters can tweaked from the command line.
Others must be configured within the network scripts themselves.

Original scripts modified from ``nvidia-examples`` scripts in `NGC
TensorFlow Container`_

Requirements
~~~~~~~~~~~~

TensorFlow
^^^^^^^^^^

::

   pip install tensorflow-gpu==1.10.0

OpenMPI
^^^^^^^

::

   wget -q -O - https://www.open-mpi.org/software/ompi/v3.0/downloads/openmpi-3.0.0.tar.gz | tar -xz
   cd openmpi-3.0.0
   ./configure --enable-orterun-prefix-by-default --with-cuda --prefix=/usr/local/mpi --disable-getpwuid
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
       echo "plm_rsh_agent = /usr/local/mpi/bin/rsh_warn.sh" >> /usr/local/mpi/etc/openmpi-mca-params.conf

Horovod
^^^^^^^

::

   export HOROVOD_GPU_ALLREDUCE=NCCL
   export HOROVOD_NCCL_INCLUDE=/usr/include
   export HOROVOD_NCCL_LIB=/usr/lib/x86_64-linux-gnu
   export HOROVOD_NCCL_LINK=SHARED
   export HOROVOD_WITHOUT_PYTORCH=1
   pip install horovod==0.15.1

.. _the original paper: https://arxiv.org/pdf/1512.03385.pdf
.. _NGC TensorFlow Container: https://www.nvidia.com/en-us/gpu-cloud/deep-learning-containers/
