# DALI Fuzzing Instruction

This is instruction how to run fuzzing on DALI pipeline.


## Setup AFL
```
sudo apt-get install clang-6.0 build-essential llvm-6.0-dev gnuplot-nox

sudo update-alternatives --install /usr/bin/clang clang `which clang-6.0` 1
sudo update-alternatives --install /usr/bin/clang++ clang++ `which clang++-6.0` 1
sudo update-alternatives --install /usr/bin/llvm-config llvm-config `which llvm-config-6.0` 1
sudo update-alternatives --install /usr/bin/llvm-symbolizer llvm-symbolizer `which llvm-symbolizer-6.0` 1

echo core | sudo tee /proc/sys/kernel/core_pattern

wget http://lcamtuf.coredump.cx/afl/releases/afl-latest.tgz
tar xvf afl-latest.tgz
cd afl-2.52b   # replace with whatever the current version is
make && make -C llvm_mode CXX=g++
make install

```


## Build DALI fuzzing targets
```
cmake -DCMAKE_CXX_COMPILER=afl-clang-fast++ -DCMAKE_C_COMPILER=afl-clang-fast -DCUDA_TARGET_ARCHS=61 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_FUZZING=ON -DBUILD_PYTHON=OFF -DBUILD_LMDB=OFF -DBUILD_NVOF=OFF ..
```

## Run fuzzing
```
afl-fuzz -i /DALI_extra/db/fuzzing/bmp/ -m none -o fuzz_results ./build/dali/python/nvidia/dali/test/dali_rn50_fuzzing_target.bin @@
```
