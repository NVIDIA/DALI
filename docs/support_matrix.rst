Supported NVIDIA hardware, CUDA, OS, and CUDA driver
====================================================

.. |compatibility link| replace:: enhanced CUDA compatibility guide
.. _compatibility link : https://docs.nvidia.com/deploy/cuda-compatibility/index.html#enhanced-compat-minor-releases
.. |PEP599 link| replace:: PEP599 - The manylinux2014 Platform Tag
.. _PEP599 link : https://www.python.org/dev/peps/pep-0599/


.. table::

  +----------------------------------+---------------+---------------------------+---------------------------------------+-------------------------+---------------------+--------------------+-------------------------------------------------+
  | Supported NVIDIA Hardware        | DALI build    | CUDA version              | Supported OS                          | CUDA Compute Capability | CUDA Driver Version | Platform           | Distribution                                    |
  +==================================+===============+===========================+=======================================+=========================+=====================+====================+=================================================+
  | - NVIDIA Hopper GPU architecture | cuda110       | 11.0 and newer,           | - Ubuntu 16.04                        | SM 3.5 and later        | r450 or later       | linux x86_64       | - Prebuilt wheel available from developers page |
  | - NVIDIA Ampere GPU architecture |               | see |compatibility link|_ | - Ubuntu 18.04                        |                         |                     |                    | - Conda can be build from source                |
  | - Turing                         |               |                           | - Ubuntu 20.04                        |                         |                     |                    |                                                 |
  | - Volta                          |               |                           | - RHEL 7                              |                         |                     |                    |                                                 |
  | - Pascal                         |               |                           | - RHEL 8                              |                         |                     |                    |                                                 |
  | - Maxwell                        |               |                           | - and other |PEP599 link|_ compatible |                         |                     |                    |                                                 |
  | - Kepler                         |               |                           |                                       |                         |                     |                    |                                                 |
  +----------------------------------+---------------+---------------------------+---------------------------------------+-------------------------+---------------------+--------------------+-------------------------------------------------+
  | - NVIDIA Ampere GPU architecture | cuda110       | 11.0 and newer,           | - Ubuntu 18.04                        | SM 7.0 and later        | r450 or later       | linux aarch64 SBSA | - Prebuilt wheel available from developers page |
  | - Turing                         |               | see |compatibility link|_ | - RHEL 8                              |                         |                     |                    |                                                 |
  | - Volta                          |               |                           | - and other |PEP599 link|_ compatible |                         |                     |                    |                                                 |
  +----------------------------------+---------------+---------------------------+---------------------------------------+-------------------------+---------------------+--------------------+-------------------------------------------------+
  | - Turing                         | cuda102       | 10.2                      | - Ubuntu 14.04                        | SM 3.5 and later        | r440 or later       | linux x86_64       | - Prebuilt wheel available from developers page |
  | - Volta                          |               |                           | - Ubuntu 16.04                        |                         |                     |                    | - Conda can be build from source                |
  | - Pascal                         |               |                           | - Ubuntu 18.04                        |                         |                     |                    |                                                 |
  | - Maxwell                        |               |                           | - RHEL 7                              |                         |                     |                    |                                                 |
  | - Kepler                         |               |                           | - and other |PEP599 link|_ compatible |                         |                     |                    |                                                 |
  +----------------------------------+---------------+---------------------------+---------------------------------------+-------------------------+---------------------+--------------------+-------------------------------------------------+
  | - Xavier                         | Not Available | 10.2                      | Jetpack 4.4                           | SM 5.3 and later        | Jetpack 4.4         | Jetpack 4.4        | - Python wheel can be build from source         |
  +----------------------------------+---------------+---------------------------+---------------------------------------+-------------------------+---------------------+--------------------+-------------------------------------------------+