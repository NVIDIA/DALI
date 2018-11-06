#include <dali/pipeline/data/backend.h>
#include "dali/test/dali_operator_test.h"

namespace dali {

class CoinFlipV2Test : public DaliOperatorTest<int, float> {


  int SetInputs() const override {
    return -1;
  }


  std::string SetOperator() const override {
    return "CoinFlip";
  }


  bool Verify(float outputs, float anticipated_outputs) const override {
    auto diff = outputs - anticipated_outputs;
    diff = diff < 0 ? -diff : diff;
    return diff < epsilon_;
  }


 private:
  float epsilon_ = 0.0001f;
};

TEST_F(CoinFlipV2Test, Always1) {
  this->RunTest({{"probability", 1.f}}, 1);
}


TEST_F(CoinFlipV2Test, Always0) {
  this->RunTest({{"probability", 0.f}}, 0);
}

}  // namespace dali