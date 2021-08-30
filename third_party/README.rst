NVIDIA DALI third party dependencies
====================================
This part of the repository contains extra dependencies required to build DALI, consisting mostly of externally hosted subrepositories:

+-----------------+---------------------+---------------------+
| Repository      | Version             | License             |
+=================+=====================+=====================+
| |benchmark|_    | |benchmarkver|_     | |benchmarklic|_     |
+-----------------+---------------------+---------------------+
| |preprocessor|_ | |preprocessorver|_  | |preprocessorlic|_  |
+-----------------+---------------------+---------------------+
| |cocoapi|_      | |cocoapiver|_       | |cocoapilic|_       |
+-----------------+---------------------+---------------------+
| |cutlass|_      | |cutlassver|_       | |cutlasslic|_       |
+-----------------+---------------------+---------------------+
| |dlpack|_       | |dlpackver|_        | |dlpacklic|_        |
+-----------------+---------------------+---------------------+
| |ffts|_         | |fftsver|_          | |fftslic|_          |
+-----------------+---------------------+---------------------+
| |fmt|_          | |fmtver|_           | |fmtlic|_           |
+-----------------+---------------------+---------------------+
| |googletest|_   | |googletestver|_    | |googletestlic|_    |
+-----------------+---------------------+---------------------+
| |pybind11|_     | |pybind11ver|_      | |pybind11lic|_      |
+-----------------+---------------------+---------------------+
| |rapidjson|_    | |rapidjsonver|_     | |rapidjsonlic|_     |
+-----------------+---------------------+---------------------+
| |rmm|_          | |rmmver|_           | |rmmlic|_           |
+-----------------+---------------------+---------------------+
| |spdlog|_       | |spdlogver|_        | |spdloglic|_        |
+-----------------+---------------------+---------------------+
| |thrust|_       | |thrustver|_        | |thrustlic|_        |
+-----------------+---------------------+---------------------+

.. |benchmark| replace:: Google Benchmark
.. _benchmark: https://github.com/google/benchmark
.. |benchmarkver| replace:: 1.5.6
.. _benchmarkver: https://github.com/google/benchmark/releases/tag/v1.5.6
.. |benchmarklic| replace:: Apache License 2.0
.. _benchmarklic: https://github.com/google/benchmark/blob/master/LICENSE

.. |preprocessor| replace:: Boost Preprocessor
.. _preprocessor: https://github.com/boostorg/preprocessor
.. |preprocessorver| replace:: 1.77.0
.. _preprocessorver: https://github.com/boostorg/preprocessor/releases/tag/boost-1.77.0
.. |preprocessorlic| replace:: Boost Software License 1.0
.. _preprocessorlic: https://github.com/boostorg/boost/blob/master/LICENSE_1_0.txt

.. |cocoapi| replace:: COCO API
.. _cocoapi: https://github.com/cocodataset/cocoapi
.. |cocoapiver| replace:: Top-of-tree (Feb 20, 2020)
.. _cocoapiver: https://github.com/cocodataset/cocoapi/tree/8c9bcc3cf640524c4c20a9c40e89cb6a2f2fa0e9
.. |cocoapilic| replace:: BSD 2-Clause License
.. _cocoapilic: https://github.com/cocodataset/cocoapi/blob/master/license.txt

.. |cutlass| replace:: CUTLASS
.. _cutlass: https://github.com/NVIDIA/cutlass
.. |cutlassver| replace:: 2.5.0
.. _cutlassver: https://github.com/NVIDIA/cutlass/releases/tag/v2.5.0
.. |cutlasslic| replace:: BSD 3-Clause License
.. _cutlasslic: https://github.com/NVIDIA/cutlass/blob/master/LICENSE.txt

.. |dlpack| replace:: DLPack
.. _dlpack: https://github.com/dmlc/dlpack
.. |dlpackver| replace:: 0.6
.. _dlpackver: https://github.com/dmlc/dlpack/releases/tag/v0.6
.. |dlpacklic| replace:: Apache License 2.0
.. _dlpacklic: https://github.com/dmlc/dlpack/blob/main/LICENSE

.. |ffts| replace:: FFTS
.. _ffts: https://github.com/JanuszL/ffts
.. |fftsver| replace:: Custom fork top-of-tree (Jan 23, 2020)
.. _fftsver: https://github.com/JanuszL/ffts/tree/c9a9f61a60505751cac385ed062ce2720bdf07d4
.. |fftslic| replace:: BSD 3-Clause License
.. _fftslic: https://github.com/JanuszL/ffts/blob/master/COPYRIGHT

.. |fmt| replace:: {fmt}
.. _fmt: https://github.com/fmtlib/fmt
.. |fmtver| replace:: 8.0.1
.. _fmtver: https://github.com/fmtlib/fmt/releases/tag/8.0.1
.. |fmtlic| replace:: MIT License
.. _fmtlic: https://github.com/fmtlib/fmt/blob/master/LICENSE.rst

.. |googletest| replace:: GoogleTest
.. _googletest: https://github.com/google/googletest
.. |googletestver| replace:: 1.11.0
.. _googletestver: https://github.com/google/googletest/releases/tag/release-1.11.0
.. |googletestlic| replace:: BSD 3-Clause License
.. _googletestlic: https://github.com/google/googletest/blob/master/LICENSE

.. |pybind11| replace:: pybind11
.. _pybind11: https://github.com/pybind/pybind11
.. |pybind11ver| replace:: 2.7.1
.. _pybind11ver: https://github.com/pybind/pybind11/releases/tag/v2.7.1
.. |pybind11lic| replace:: BSD 3-Clause License
.. _pybind11lic: https://github.com/pybind/pybind11/blob/master/LICENSE

.. |rapidjson| replace:: RapidJSON
.. _rapidjson: https://github.com/Tencent/rapidjson
.. |rapidjsonver| replace:: Top-of-tree (Jul 9, 2021)
.. _rapidjsonver: https://github.com/Tencent/rapidjson/tree/48fbd8cd202ca54031fe799db2ad44ffa8e77c13
.. |rapidjsonlic| replace:: MIT License, BSD 3-Clause License, JSON License
.. _rapidjsonlic: https://github.com/Tencent/rapidjson/blob/master/license.txt

.. |rmm| replace:: RMM
.. _rmm: https://github.com/mzient/rmm
.. |rmmver| replace:: Custom fork top-of-tree (Mar 1, 2021)
.. _rmmver: https://github.com/mzient/rmm/tree/ae8cecc6b006d9d0e49f972b47db37f965f6da16
.. |rmmlic| replace:: Apache License 2.0
.. _rmmlic: https://github.com/mzient/rmm/blob/branch-0.18/LICENSE

.. |spdlog| replace:: spdlog
.. _spdlog: https://github.com/gabime/spdlog
.. |spdlogver| replace:: 1.9.0
.. _spdlogver: https://github.com/gabime/spdlog/releases/tag/v1.9.0
.. |spdloglic| replace:: MIT License
.. _spdloglic: https://github.com/gabime/spdlog/blob/v1.x/LICENSE

.. |thrust| replace:: Thrust
.. _thrust: https://github.com/NVIDIA/thrust
.. |thrustver| replace:: 1.13.0
.. _thrustver: https://github.com/NVIDIA/thrust/releases/tag/1.13.0
.. |thrustlic| replace:: Apache License 2.0
.. _thrustlic: https://github.com/NVIDIA/thrust/blob/main/LICENSE
