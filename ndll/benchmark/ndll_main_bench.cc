#include <benchmark/benchmark.h>

#include "ndll/common.h"

namespace ndll {

static void test(benchmark::State &state) {
  while (state.KeepRunning())
    cout << "ran again" << endl;
}
BENCHMARK(test)->Iterations(10);

}

BENCHMARK_MAIN()
