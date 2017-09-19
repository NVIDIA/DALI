# Build
run `make` to build the static lib & tests. The lib depends only on cuda and NPP. Building the tests requires google c++ testing framework. This is simple to install, see [gtest repo](https://github.com/google/googletest/blob/master/googletest/README.md). You will need to update the `GTEST_DIR` variable in `makefile` to the installation path.

If neccessary, edit the `CUDA_DIR` variable in `makefile` to point to the root of your cuda installation.

This code only supports >=SM3
