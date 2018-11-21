#include <dali/pipeline/data/backend.h>
#include "dali/test/dali_operator_test.h"

namespace dali {

class CoinFlipV2Test : public DaliOperatorTest<int, int> {


  /// No-op for this operator
  std::vector<std::pair<int, Shape>> SetInputs() const override {
    return {};
  }


  std::string SetOperator() const override {
    return "CoinFlip";
  }


  bool Verify(int outputs, int anticipated_outputs) const override {
    return outputs == anticipated_outputs;
  }

};

TEST_F(CoinFlipV2Test, Always1) {
  this->RunTest<CPUBackend>({{"probability", 1.f}}, 1);
}


TEST_F(CoinFlipV2Test, Always0) {
  this->RunTest<CPUBackend>({{"probability", 0.f}}, 0);
}

}  // namespace dali