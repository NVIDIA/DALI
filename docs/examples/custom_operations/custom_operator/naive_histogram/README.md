# DALI custom operator example

This is an example/template of a DALI custom operator, packaged as a DALI plugin.
The implemented operation is a histogram of a grayscale image.
The `NaiveHistogram` operator is a GPU-only operator, which uses a suboptimal
CUDA kernel to calculate the histogram. The purpose here is not to showcase
how to write CUDA kernels, but to present how to use CUDA kernels within
DALI operators.

The example shows:

* a GPU operator with a scalar argument (`n_bins`) read from the `OpSpec`,
* how to describe the shape and type of the output in `SetupImpl`,
* how to launch a CUDA kernel on the stream provided by DALI in `RunImpl`,
* how to build the plugin with CMake,
* how to package the plugin as a Python source distribution (sdist).

For a step-by-step introduction to writing custom operators, see the
[Create a Custom Operator in C++](https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/examples/custom_operations/custom_operator/create_a_custom_operator.html)
tutorial.

> **Important:** A plugin can only be loaded by the exact DALI version that it was built against.
> Every time you upgrade (or downgrade) DALI, or switch to a different DALI flavor (for example
> from CUDA 12 to CUDA 13 build, or to a nightly build), you must rebuild the plugin.

## Requirements

* [DALI](https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/installation.html)
* CUDA Toolkit matching the CUDA version of the installed DALI flavor
  (for example CUDA 13.x for `nvidia-dali-cuda130`)
* C++20 compiler
* CMake 3.25.2 or newer

## Building with CMake

All the commands below are run from the `naive_histogram` directory:

```bash
cmake -S . -B build
cmake --build build -j
```

This produces `build/libnaivehistogram.so`, which can be loaded with
`nvidia.dali.plugin_manager.load_library`.

By default, the CUDA code is compiled for the `80;90` architectures. To target a different GPU, pass
`-DCMAKE_CUDA_ARCHITECTURES=<arch>` to `cmake` (for example, `native` targets the GPUs of the build
machine), or set the `CUDAARCHS` environment variable.

## Running the example

1. The test needs some images. It can really be any image, as long as it's not broken.
   The test provides a preliminary image list as an example, which uses images from the
   [DALI_extra](https://github.com/NVIDIA/DALI_extra) repository, but it's not obligatory.
   Should you want to use DALI_extra too, please clone it, following
   [the instructions provided](https://github.com/NVIDIA/DALI_extra#usage) (it requires Git LFS),
   and set an environment variable defining the path to it:

   ```bash
   git clone https://github.com/NVIDIA/DALI_extra.git /path/to/dali/extra
   export DALI_EXTRA_PATH=/path/to/dali/extra
   ```

2. Run the provided example (your output should look like below):

   ```bash
   python test_naive_histogram.py
   [[11355 10555 10499 10724 10687 11213 11388 11474 11715 11407 11291 11093
     10757 10481 10547 11177 10081 10353 10380 10691 10947 10851 10872 10582]
    [44322 44408 45633 49539 53415 46655 46081 44979 42273 41195 43601 43466
     42768 43041 42755 43519 44542 47158 50718 49510 45163 44758 45982 46359]]
   ```

## Packaging the plugin as a Python package

Distributing a prebuilt `.so` file is inconvenient, because it works only with the DALI version it
was built against. Instead, the plugin can be distributed as a Python source distribution that is
compiled at installation time against the DALI version installed in the target environment.
This is the same approach that DALI uses for its TensorFlow and video plugins.

Unlike the DALI TensorFlow plugin, the example does not list DALI in its dependencies: the name of
the DALI distribution depends on its flavor (for example `nvidia-dali-cuda120` or
`nvidia-dali-nightly-cuda130`), and DALI built from source has none. Instead, the package checks the
exact DALI version and flavor when it is imported.

The packaging consists of:

* [`pyproject.toml`](pyproject.toml) - the package metadata. The package is built with
  [scikit-build-core](https://scikit-build-core.readthedocs.io), which runs CMake.
* The `install` rules at the end of [`CMakeLists.txt`](CMakeLists.txt), which put the plugin library
  and a `_build_info.py` file, recording the DALI version and CUDA version (flavor) used for the
  build, into the package.
* [`dali_naive_histogram/__init__.py`](dali_naive_histogram/__init__.py) - importing the package
  checks that the installed DALI version and flavor match the ones that the plugin was built against
  and loads the plugin.

To create the source distribution:

```bash
pip install build
python -m build --sdist
```

To install it, with DALI already installed in the environment:

```bash
pip install scikit-build-core cmake
pip install --no-build-isolation dist/dali_naive_histogram-0.1.0.tar.gz
```

The `--no-build-isolation` flag is necessary: without it, pip builds the package in an isolated
environment that does not contain the DALI installation the plugin has to be built against.

To compile the CUDA code for different GPU architectures, pass the CMake option through pip, or set
the `CUDAARCHS` environment variable:

```bash
pip install --no-build-isolation -C cmake.define.CMAKE_CUDA_ARCHITECTURES=native dist/dali_naive_histogram-0.1.0.tar.gz
```

After the installation, importing the package registers the operator. Run Python from outside of
the `naive_histogram` directory: otherwise, Python imports the `dali_naive_histogram` sources
from the current directory, which do not contain the built plugin, instead of the installed package.

```python
import dali_naive_histogram  # noqa: F401
from nvidia.dali import fn

fn.naive_histogram
```

When DALI is upgraded, rebuild the plugin. Make sure that pip does not reuse a wheel that it
cached when installing the plugin for the previous DALI version:

```bash
pip install --no-build-isolation --no-cache-dir --force-reinstall dist/dali_naive_histogram-0.1.0.tar.gz
```

## Troubleshooting

See the [Troubleshooting](https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/examples/custom_operations/custom_operator/create_a_custom_operator.html#Troubleshooting)
section of the custom operator tutorial.
