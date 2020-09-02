# DALI Fuzzing Instruction

This is instruction how to run fuzzing on DALI.

The goal of fuzzing is to find bugs in software. Fuzzer runs given binary multiple times with different inputs to look for possible problems. It generates inputs as it goes based on feedback from the tested binary. This feedback includes execution paths, previously seen errors etc. This gives better results than random search as space of possible inputs may be huge.

As a tool to run fuzzing we use [American Fuzzy Lop](https://github.com/google/AFL).


## Setup AFL
First we need to setup AFL. Script below shows how to do it on clean Ubuntu 18 installation.
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

We need to build DALI with AFL compiler extensions to enable AFL to trace the binary we want to fuzz later.
```
cmake -DCMAKE_CXX_COMPILER=afl-clang-fast++ -DCMAKE_C_COMPILER=afl-clang-fast -DCUDA_TARGET_ARCHS=61 -DCMAKE_EXPORT_COMPILE_COMMANDS=ON -DBUILD_FUZZING=ON -DBUILD_PYTHON=OFF -DBUILD_LMDB=OFF -DBUILD_NVOF=OFF ..
```

## Run fuzzing

Now we are ready to run fuzzing. Fuzzer takes a binary and runs it multiple times with different inputs. 

With *-i* parameter we point AFL to a directory with examples of possible inputs to start fuzzing process. Important notice, these examples should not trigger errors or exceptions.

*-m none* lifts the memory limit from tested binary.

*-o* points to the directory where AFL will write results including inputs that caused an error. This allows to reproduce them later.

At the end of the call we pass path to binary to be tested. We use *@@* to mark the place in the call where AFL should put a path to the generated input. 

```
afl-fuzz -i /DALI_extra/db/fuzzing/bmp/ -m none -o fuzz_results ./build/dali/python/nvidia/dali/test/dali_rn50_fuzzing_target.bin @@
```

When run properly, after some setup output should look similar to:

```
            american fuzzy lop 2.52b (dali_rn50_fuzzing_target.bin)

┌─ process timing ─────────────────────────────────────┬─ overall results ─────┐
│        run time : 0 days, 0 hrs, 1 min, 9 sec        │  cycles done : 0      │
│   last new path : none seen yet                      │  total paths : 4      │
│ last uniq crash : none seen yet                      │ uniq crashes : 0      │
│  last uniq hang : none seen yet                      │   uniq hangs : 0      │
├─ cycle progress ────────────────────┬─ map coverage ─┴───────────────────────┤
│  now processing : 2 (50.00%)        │    map density : 10.86% / 10.86%       │
│ paths timed out : 1 (25.00%)        │ count coverage : 1.00 bits/tuple       │
├─ stage progress ────────────────────┼─ findings in depth ────────────────────┤
│  now trying : trim 16/16            │ favored paths : 2 (50.00%)             │
│ stage execs : 25/47 (53.19%)        │  new edges on : 4 (100.00%)            │
│ total execs : 123                   │ total crashes : 0 (0 unique)           │
│  exec speed : 0.00/sec (zzzz...)    │  total tmouts : 0 (0 unique)           │
├─ fuzzing strategy yields ───────────┴───────────────┬─ path geometry ────────┤
│   bit flips : 0/0, 0/0, 0/0                         │    levels : 1          │
│  byte flips : 0/0, 0/0, 0/0                         │   pending : 4          │
│ arithmetics : 0/0, 0/0, 0/0                         │  pend fav : 2          │
│  known ints : 0/0, 0/0, 0/0                         │ own finds : 0          │
│  dictionary : 0/0, 0/0, 0/0                         │  imported : n/a        │
│       havoc : 0/0, 0/0                              │ stability : 99.80%     │
│        trim : 0.00%/60, n/a                         ├────────────────────────┘
└─────────────────────────────────────────────────────┘          [cpu000: 14%]

```
