#include "dali_operator_test.h"

namespace dali {

namespace testing {

struct InputData {
  std::vector<int> data;
  std::vector<size_t> shape;
};

std::vector<InputData> input_data;

class MyOperatorTest : public DaliOperatorTest<int, float> {
  std::vector<TensorAdapter<int>> SetInputs() const noexcept override {
    std::vector<TensorAdapter<int>> ret;
    for (auto in : input_data) {
      ret.emplace_back(in.data, in.shape);
    }
    return ret;
  }


  OperatorsGraph SetOperators() const noexcept override {
    return "MyOp";
  };


  bool Verify(TensorAdapter<float> output, TensorAdapter<float> anticipated_output) const noexcept override {
    return false;
  }
};

TEST_F(MyOperatorTest, MyOperatorTestCase) {
  std::vector<TensorAdapter<float>> anticipated_outputs;
  this->RunTest({{"arg1", true},
                 {"arg2", 5.0}}, anticipated_outputs);
}

} // namespace testing
} // namespace dali