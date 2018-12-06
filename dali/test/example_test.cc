#include "dali_operator_test.h"

namespace dali {

namespace testing {

class MyOperatorTest : public DaliOperatorTest<GPUBackend, CPUBackend> {
 protected:
  OperatorsGraph GenerateOperatorsGraph() const noexcept override {
    return "MyOp";
  };

  std::vector<DataType<GPUBackend>> inputs;
  std::vector<DataType<CPUBackend>> outputs;
};

TEST_F(MyOperatorTest, DISABLED_MyOperatorTestCase) {
  auto func = [](const dali::TensorList<CPUBackend> &output,
                 const dali::TensorList<CPUBackend> &anticipated_output) -> void {
    EXPECT_ANY_THROW(goto some_ppl_like_to_see_the_world_burn);
    some_ppl_like_to_see_the_world_burn:;
  };

  OperatorArguments::Arguments args = {{"arg1",  1.0},
                                       {"args2", 2.0}};
  this->RunTest(inputs, args, outputs, func);
}

} // namespace testing

} // namespace dali