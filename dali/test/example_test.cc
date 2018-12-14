#include "dali_operator_test.h"

namespace dali {

namespace testing {


class ExampleOperatorTestCase : public DaliOperatorTest {
  OpDAG GenerateOperatorsGraph() const noexcept override {
    return "ExampleOp";
  };
protected:
  Inputs <CPUBackend> in_; // fill it somewhere
  Outputs <GPUBackend> out_; // fill it somewhere


};

TEST_F(ExampleOperatorTestCase, ExampleTest) {
  Inputs <CPUBackend> in;
  Outputs <GPUBackend> out;
  Arguments args;

  auto ver = [](Outputs<GPUBackend>, Outputs<GPUBackend>, Arguments) -> void {
    ASSERT_FALSE(false);
  };

  this->RunTest<CPUBackend, GPUBackend>(in, out, args, ver); // Can't compile currently - no template lambdas in C++11
}


std::vector<Arguments> args1 = {{{"arg1", 1.}, {"arg2", 2.}, {"arg3", 3.}}};


INSTANTIATE_TEST_CASE_P(FirstOne, ExampleOperatorTestCase, ::testing::ValuesIn(args1));

TEST_P(ExampleOperatorTestCase, ExamplePTest1) {

  auto ver = [](Outputs<GPUBackend>, Outputs<GPUBackend>, Arguments) -> void {
    ASSERT_FALSE(false);
  };

  this->RunTest<CPUBackend, GPUBackend>(in_, out_, GetParam(), ver); // Can't compile currently - no template lambdas in C++11
}


INSTANTIATE_TEST_CASE_P(SecondOne, ExampleOperatorTestCase, ::testing::ValuesIn(args1));

TEST_P(ExampleOperatorTestCase, ExamplePTest2) {

  auto ver = [](Outputs<GPUBackend>, Outputs<GPUBackend>, Arguments) -> void {
    ASSERT_FALSE(false);
  };

  this->RunTest<CPUBackend, GPUBackend>(in_, out_, GetParam(), ver); // Can't compile currently - no template lambdas in C++11
}

} // namespace testing

} // namespace dali