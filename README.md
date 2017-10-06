# Documentation
The code is heavily documented. Run `doxygen Doxyfile` to build the documentation.

# Build
`mkdir build && cd build && cmake -DCMAKE_BUILD_TYPE=Release -DUSE_NVTX=OFF -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON && make -j 20`

Note: NDLL has submodules (gtest & google benchmark). Use `--recursive` when cloning the repo.