#include "ndll/pipeline/data_reader_op.h"

#include <gtest/gtest.h>

#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/op_spec.h"
#include "ndll/test/ndll_test.h"

#include <cstdio>

namespace ndll {

template <typename Backend>
class DummyDataReaderOp : public DataReader<Backend> {
 public:
  DummyDataReaderOp(const OpSpec &spec) : DataReader<Backend>(spec) {

  }

  void Prefetch() {
    static int i = -1;
    i++;
    printf("prefetched %d\n", i);
  }
};

template <typename Backend>
class PrefetchedDataReaderTest : public NDLLTest {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

typedef ::testing::Types<CPUBackend> TestTypes;

TYPED_TEST_CASE(PrefetchedDataReaderTest, TestTypes);

TYPED_TEST(PrefetchedDataReaderTest, test) {
  return;
}

};  // namespace ndll
