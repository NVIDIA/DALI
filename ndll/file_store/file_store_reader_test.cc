#include "ndll/file_store/file_store_reader.h"

#include <gtest/gtest.h>

#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/op_spec.h"
#include "ndll/test/ndll_test.h"

namespace ndll {

template <typename Backend>
class FileStoreReaderTest : public NDLLTest {
 public:
  void SetUp() override {}
  void TearDown() override {}
};

typedef ::testing::Types<CPUBackend> TestTypes;

TYPED_TEST_CASE(FileStoreReaderTest, TestTypes);

TYPED_TEST(FileStoreReaderTest, test) {
  return;
}

}; // namespace ndll
