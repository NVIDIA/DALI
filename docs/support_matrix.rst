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
  | - NVIDIA Ampere GPU architecture | cuda110       | 11.0 and newer,           | - Ubuntu 16.04                        | SM 3.5 and later        | r450 or later       | linux x86_64       | - Prebuild wheel available from developers page |
  | - Turing                         |               | see |compatibility link|_ | - Ubuntu 18.04                        |                         |                     |                    | - Conda can be build from source                |
  | - Volta                          |               |                           | - Ubuntu 20.04                        |                         |                     |                    |                                                 |
  | - Pascal                         |               |                           | - RHEL 7                              |                         |                     |                    |                                                 |
  | - Maxwell                        |               |                           | - RHEL 8                              |                         |                     |                    |                                                 |
  | - Kepler                         |               |                           | - and other |PEP599 link|_ compatible |                         |                     |                    |                                                 |
  +----------------------------------+---------------+---------------------------+---------------------------------------+-------------------------+---------------------+--------------------+-------------------------------------------------+
  | - NVIDIA Ampere GPU architecture | cuda110       | 11.0 and newer,           | - Ubuntu 18.04                        | SM 7.0 and later        | r450 or later       | linux aarch64 SBSA | - Prebuild wheel available from developers page |
  | - Turing                         |               | see |compatibility link|_ | - RHEL 8                              |                         |                     |                    |                                                 |
  | - Volta                          |               |                           | - and other |PEP599 link|_ compatible |                         |                     |                    |                                                 |
  +----------------------------------+---------------+---------------------------+---------------------------------------+-------------------------+---------------------+--------------------+-------------------------------------------------+
  | - Turing                         | cuda100       | 10.0                      | - Ubuntu 14.04                        | SM 3.5 and later        | r410 or later       | linux x86_64       | - Prebuild wheel available from developers page |
  | - Volta                          |               |                           | - Ubuntu 16.04                        |                         |                     |                    | - Conda can be build from source                |
  | - Pascal                         |               |                           | - Ubuntu 18.04                        |                         |                     |                    |                                                 |
  | - Maxwell                        |               |                           | - RHEL 7                              |                         |                     |                    |                                                 |
  | - Kepler                         |               |                           | - and other |PEP599 link|_ compatible |                         |                     |                    |                                                 |
  +----------------------------------+---------------+---------------------------+---------------------------------------+-------------------------+---------------------+--------------------+-------------------------------------------------+
  | - Xavier                         | Not Available | 10.2                      | Jetpack 4.4                           | SM 5.3 and later        | Jetpack 4.4         | Jetpack 4.4        | - Python wheel can be build from source         |
  +----------------------------------+---------------+---------------------------+---------------------------------------+-------------------------+---------------------+--------------------+-------------------------------------------------+
  | - Xavier                         | Not Available | 10.2                      | DRIVE OS 5.1.0.0                      | SM 5.3 and later        | DRIVE OS 5.1.0.0    | DRIVE OS 5.1.0.0   | - Native library can be build from source       |
  +----------------------------------+---------------+---------------------------+---------------------------------------+-------------------------+---------------------+--------------------+-------------------------------------------------+
