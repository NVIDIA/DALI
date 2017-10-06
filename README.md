# Documentation
The code is heavily documented. Run `doxygen Doxyfile` to build the documentation.

# Build
`mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DUSE_NVTX=OFF -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON && make -j 20`

Note: NDLL has submodules (gtest & google benchmark). Use `--recursive` when cloning the repo.

# Results
See experimental NDLL+Caffe2 integration [here.](https://nvdl.githost.io/dgx/caffe2/tree/17.10-devel-ndll)

Standalone Data Loader Performance:
![data-loader-perf](docs/results/c2-ndll-standalone.png)

ResNet-50 Performance:
![rn50-perf](docs/results/c2-ndll-rn50.png)

ResNet-18 Performance:
![rn18-perf](docs/results/c2-ndll-rn18.png)

Notes:
- Hybrid jpeg decode is very fast but has significant overhead, mostly due to cudaLaunch latency (>50% of the overhead w/ 32 batch size). This can be reduced significantly by moving to cudaLaunchKernel to launch non-batched kernels, and by eventually replacing the single-image resize & yuv->rgb kernels with batched version.
- NDLL currently pre-sizes buffers to avoid synchronous memory allocations and slow startup time. We're looking into moving to caching allocators for host & device to remove the need for this and avoid wasteful memory allocation.
- NDLL Hybrid decode uses significantly more GPU memory than the normal CPU pipeline. This can be reduced with some memory optimizations in the Pipeline class.