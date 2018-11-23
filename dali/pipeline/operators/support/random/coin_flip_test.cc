#include <dali/pipeline/data/backend.h>
#include "dali/test/dali_operator_test.h"

namespace dali {

/// This test is DISABLED, since (1) it was used for creating test POC and
/// (2) testing pipeline does not support SupportBackend yet.
class DISABLED_CoinFlipV2Test : public DaliOperatorTest<int, int> {


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

TEST_F(DISABLED_CoinFlipV2Test, Always1) {
//  this->RunTest<SupportBackend>({{"probability", 1.f}}, 1);
}


TEST_F(DISABLED_CoinFlipV2Test, Always0) {
//  this->RunTest<SupportBackend>({{"probability", 0.f}}, 0);
}

}  // namespace dali