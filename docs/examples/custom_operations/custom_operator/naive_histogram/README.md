# DALI custom operator example

This is an example/template of a DALI Custom operator.
The implemented operation is a histogram on a grayscale image.
The NaiveHistogram operator is a GPU-only operator, which uses a suboptimal
CUDA kernel to calculate the histogram. The purpose here is not to show-case
how the write CUDA kernels, but to present how to use CUDA kernels within
DALI operators.

# Running the example

To run the example, please follow the steps:

1. Clone or download the repo
1. Install DALI wheel, for example using [these instructions](https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/installation.html#pip-official-releases)
1. Build the `libnaivehistogram.so`:
```bash
$ cd naive_histogram
$ mkdir build
$ cd build
$ cmake ..
$ make -j
```

4. To run the example, you need to provide some test files (actually, images). It can 
really be any image, as long as it's not broken. The test provides preliminary image list as an 
example. We're using a [DALI_Extra](https://github.com/NVIDIA/DALI_extra) repository for that
purpose, but it's not obligatory. Should you want to use DALI Extra too, please clone (or download)
the repo. Be sure to use [the instructions provided](https://github.com/NVIDIA/DALI_Extra#usage).
After cloning, please set an environment variable defining the path to DALI Extra:
```bash
git clone -d /path/to/dali/extra https://github.com/NVIDIA/DALI_extra
export DALI_EXTRA_PATH=/path/to/dali/extra
```

5. Run the provided example (your output should look like below):
```bash
$ cd naive_histogram
$ python naive_histogram_test.py
[[11355 10555 10499 10724 10687 11213 11388 11474 11715 11407 11291 11093
  10757 10481 10547 11177 10081 10353 10380 10691 10947 10851 10872 10582]
 [44322 44408 45633 49539 53415 46655 46081 44979 42273 41195 43601 43466
  42768 43041 42755 43519 44542 47158 50718 49510 45163 44758 45982 46359]]
```

# More information

More information about DALI Custom Operator you can find in [DALI Documentation](https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/examples/custom_operations/custom_operator/create_a_custom_operator.html)

