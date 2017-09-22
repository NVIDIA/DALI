#include "ndll/pipeline/operators/hybrid_decoder.h"

#include <gtest/gtest.h>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/pipeline.h"
#include "ndll/test/ndll_main_test.h"

namespace ndll {

namespace {
// const vector<string> jpegs_to_test = {img_folder + "};
} // namespace

template <typename Color>
class HybridDecoderTest : public NDLLTest {
public:
protected:
  bool color_;
  int C_;
  
  vector<uint8*> images_;
};

// TYPED_TEST(HybridDecoderTest, TestDecodeJPEGS) {

// }

} // namespace ndll
