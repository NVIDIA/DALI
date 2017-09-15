#include "ndll/pipeline/pipeline.h"

#include <benchmark/benchmark.h>

namespace ndll {

static void test2(benchmark::State &state) {
  while (state.KeepRunning())
    cout << "ran again2" << endl;
}
BENCHMARK(test2)->Iterations(10);

} // namespace ndll
