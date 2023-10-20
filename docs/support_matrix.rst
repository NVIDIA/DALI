Supported NVIDIA hardware, CUDA, OS, and CUDA driver
====================================================

.. |compatibility link| replace:: enhanced CUDA compatibility guide
.. _compatibility link : https://docs.nvidia.com/deploy/cuda-compatibility/index.html#enhanced-compat-minor-releases
.. |PEP599 link| replace:: PEP599 - The manylinux2014 Platform Tag
.. _PEP599 link : https://www.python.org/dev/peps/pep-0599/


.. table::

  +----------------------------------+---------------+---------------------------+---------------------------------------+-------------------------+---------------------+--------------------+---------------------------------------------------------------+
  | Supported NVIDIA Hardware        | DALI build    | CUDA version              | Supported OS                          | CUDA Compute Capability | CUDA Driver Version | Platform           | Distribution                                                  |
  +==================================+===============+===========================+=======================================+=========================+=====================+====================+===============================================================+
  | - NVIDIA Hopper GPU architecture | cuda120       | 12.0 and newer,           | - Ubuntu 16.04                        | SM 5.0 and later        | r525 or later       | linux x86_64       | - :ref:`Prebuilt wheel available <pip wheels>`                |
  | - NVIDIA Ampere GPU architecture |               | see |compatibility link|_ | - Ubuntu 18.04                        |                         |                     |                    | - Conda can be build from source                              |
  | - Turing                         |               |                           | - Ubuntu 20.04                        |                         |                     |                    |                                                               |
  | - Volta                          |               |                           | - RHEL 7                              |                         |                     |                    |                                                               |
  | - Pascal                         |               |                           | - RHEL 8                              |                         |                     |                    |                                                               |
  | - Maxwell                        |               |                           | - and other |PEP599 link|_ compatible |                         |                     |                    |                                                               |
  +----------------------------------+---------------+---------------------------+---------------------------------------+-------------------------+---------------------+--------------------+---------------------------------------------------------------+
  | - NVIDIA Ampere GPU architecture | cuda120       | 12.0 and newer,           | - Ubuntu 18.04                        | SM 7.0 and later        | r525 or later       | linux aarch64 SBSA | - :ref:`Prebuilt wheel available <pip wheels>`                |
  | - Turing                         |               | see |compatibility link|_ | - RHEL 8                              |                         |                     |                    |                                                               |
  | - Volta                          |               |                           | - and other |PEP599 link|_ compatible |                         |                     |                    |                                                               |
  +----------------------------------+---------------+---------------------------+---------------------------------------+-------------------------+---------------------+--------------------+---------------------------------------------------------------+
  | - NVIDIA Hopper GPU architecture | cuda110       | 11.0 and newer,           | - Ubuntu 16.04                        | SM 3.5 and later        | r450 or later       | linux x86_64       | - :ref:`Prebuilt wheel available <pip wheels>`                |
  | - NVIDIA Ampere GPU architecture |               | see |compatibility link|_ | - Ubuntu 18.04                        |                         |                     |                    | - Conda can be build from source                              |
  | - Turing                         |               |                           | - Ubuntu 20.04                        |                         |                     |                    |                                                               |
  | - Volta                          |               |                           | - RHEL 7                              |                         |                     |                    |                                                               |
  | - Pascal                         |               |                           | - RHEL 8                              |                         |                     |                    |                                                               |
  | - Maxwell                        |               |                           | - and other |PEP599 link|_ compatible |                         |                     |                    |                                                               |
  | - Kepler                         |               |                           |                                       |                         |                     |                    |                                                               |
  +----------------------------------+---------------+---------------------------+---------------------------------------+-------------------------+---------------------+--------------------+---------------------------------------------------------------+
  | - NVIDIA Ampere GPU architecture | cuda110       | 11.0 and newer,           | - Ubuntu 18.04                        | SM 7.0 and later        | r450 or later       | linux aarch64 SBSA | - :ref:`Prebuilt wheel available <pip wheels>`                |
  | - Turing                         |               | see |compatibility link|_ | - RHEL 8                              |                         |                     |                    |                                                               |
  | - Volta                          |               |                           | - and other |PEP599 link|_ compatible |                         |                     |                    |                                                               |
  +----------------------------------+---------------+---------------------------+---------------------------------------+-------------------------+---------------------+--------------------+---------------------------------------------------------------+
  | - Xavier                         | Not Available | 11.8                      | Jetpack 5.0.2                         | SM 5.3 and later        | Jetpack 5.0.2       | Jetpack 5.0.2      | - :ref:`Python wheel can be build from source <jetson build>` |
  +----------------------------------+---------------+---------------------------+---------------------------------------+-------------------------+---------------------+--------------------+---------------------------------------------------------------+
