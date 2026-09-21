.. _running_examples:

Running the Examples
====================

All tutorials in this section are generated from Jupyter notebooks stored in the
:fileref:`docs/examples` directory of the DALI repository. Every tutorial page starts with a note
linking to its source notebook at the revision matching this version of the documentation, so you
can download it, modify it and run it yourself.

Running Locally
---------------

1. :doc:`Install DALI <../installation>`.
2. Clone the DALI repository (or download the notebook you are interested in from the link at the
   top of the tutorial) and install Jupyter::

      git clone https://github.com/NVIDIA/DALI.git
      pip install jupyter

3. Most of the notebooks read their input data from the
   `DALI_extra <https://github.com/NVIDIA/DALI_extra>`_ repository (``git-lfs`` is required to
   clone it). Check out the revision listed in :fileref:`DALI_EXTRA_VERSION` and point the
   ``DALI_EXTRA_PATH`` environment variable at it::

      git clone https://github.com/NVIDIA/DALI_extra.git
      git -C DALI_extra checkout $(cat DALI/DALI_EXTRA_VERSION)
      export DALI_EXTRA_PATH=$(pwd)/DALI_extra

4. Start Jupyter in ``DALI/docs/examples`` and open the notebook::

      cd DALI/docs/examples
      jupyter notebook

Running in Google Colab
-----------------------

Every tutorial page has an *Open in Colab* link that opens the notebook straight from GitHub in
`Google Colab <https://colab.research.google.com>`_. Make sure that a GPU runtime is selected
(*Runtime -> Change runtime type -> Hardware accelerator: GPU*) and, before running the tutorial,
execute the following cell to install DALI, fetch the test data and export ``DALI_EXTRA_PATH``::

   !curl -sSL https://raw.githubusercontent.com/NVIDIA/DALI/main/docs/examples/colab_setup.py -o colab_setup.py
   %run colab_setup.py

The :fileref:`docs/examples/colab_setup.py` script installs the latest DALI release for the CUDA
version available in the runtime and checks out the DALI_extra revision matching the ``main``
branch. Use ``--ref <tag or commit>`` to match the notebooks of a particular DALI release
(for example ``%run colab_setup.py --ref v2.3.0``), ``--nightly`` to install the nightly build
when running notebooks from ``main``, or ``%run colab_setup.py --help`` to list all the options.

.. note::

   Some tutorials (for example the custom operator ones) depend on additional files stored next
   to the notebook. When running them in Colab, download those files from the
   :fileref:`docs/examples` directory into the Colab working directory first.
