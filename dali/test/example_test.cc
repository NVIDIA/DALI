#include "dali_operator_test.h"

namespace dali {

namespace testing {

void ExVerify(DaliOperatorTest::Outputs<GPUBackend>, DaliOperatorTest::Outputs<GPUBackend>,
              Arguments) { ASSERT_TRUE(false); };

class ExampleOperatorTestCase : public DaliOperatorTest {
  OperatorGraph GenerateOperatorsGraph() const noexcept override {
    return "ExampleOp";
  };
 protected:
  Inputs <CPUBackend> in_;
  Outputs <GPUBackend> out_;


};

TEST_F(ExampleOperatorTestCase, DISABLED_ExampleTest) {
  Inputs <CPUBackend> in;
  Outputs <GPUBackend> out;
  Arguments args;

  this->RunTest(in, out, args, [](Outputs <GPUBackend>, Outputs <GPUBackend>, Arguments) -> void {
      ASSERT_FALSE(false);
  });
}


std::vector<Arguments> args1 = {{{"arg1", 1.}, {"arg2", 2.}, {"arg3", 3.}}};


INSTANTIATE_TEST_CASE_P(FirstOne, ExampleOperatorTestCase, ::testing::ValuesIn(args1));

TEST_P(ExampleOperatorTestCase, DISABLED_ExamplePTest1) {

  Verify <GPUBackend> v = ExVerify;
  this->RunTest(in_, out_, GetParam(), v);
}


INSTANTIATE_TEST_CASE_P(SecondOne, ExampleOperatorTestCase, ::testing::ValuesIn(args1));

TEST_P(ExampleOperatorTestCase, DISABLED_ExamplePTest2) {

  Verify <GPUBackend> v = ExVerify;
  this->RunTest(in_, out_, GetParam(), v);
}

} // namespace testing

} // namespace dali