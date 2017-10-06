# Documentation
The code is heavily documented. Run `doxygen Doxyfile` to build the documentation.

# Build
`mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DUSE_NVTX=OFF -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON && make -j 20`

Note: NDLL has submodules (gtest & google benchmark). Use `--recursive` when cloning the repo.

# Results
[Experimental NDLL+Caffe2 Integration](https://nvdl.githost.io/dgx/caffe2/tree/17.10-devel-ndll)

Standalone Data Loader Performance:
![data-loader-perf](docs/results/c2-ndll-standalone.png)

ResNet-50 Performance (4GPU):
![rn50-perf](docs/results/c2-ndll-rn50.png)

ResNet-18 Performance (4GPU):
![rn18-perf](docs/results/c2-ndll-rn18.png)