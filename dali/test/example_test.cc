#include "dali_operator_test.h"

namespace dali {

namespace testing {


class MyOperatorTest : public DaliOperatorTest<GPUBackend, CPUBackend> {
  std::vector<std::unique_ptr<dali::TensorList<GPUBackend>>>
  GenerateInputs() const noexcept override {
    std::vector<std::unique_ptr<dali::TensorList<GPUBackend>>> ret;
    return ret;
  }


  OperatorsGraph GenerateOperatorsGraph() const noexcept override {
    return "MyOp";
  };


  void Verify(const dali::TensorList<CPUBackend> &output,
              const dali::TensorList<CPUBackend> &anticipated_output) const noexcept override {
    ASSERT_TRUE(false);
  }
};

TEST_F(MyOperatorTest, MyOperatorTestCase) {
  std::vector<std::unique_ptr<dali::TensorList<CPUBackend>>> anticipated_outputs;
  OperatorArguments::Arguments args = {{"arg1",  1.0},
                                       {"args2", 2.0}};
  this->RunTest(args, std::move(anticipated_outputs));
}

} // namespace testing
} // namespace dali