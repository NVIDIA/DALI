#include "dali_operator_test.h"

namespace dali {

namespace testing {


class ExampleOperatorTestCase : public DaliOperatorTest {
  std::unique_ptr<OpDag> GenerateOperatorsGraph() const noexcept override {
    return std::unique_ptr<OpDag>(new OpDagStub("ExampleOp"));
  };
protected:
  TensorListWrapper in_; // fill it somewhere
  TensorListWrapper out_; // fill it somewhere


};

TEST_F(ExampleOperatorTestCase, ExampleTest) {
  TensorListWrapper in;
  TensorListWrapper out;
  Arguments args;

  auto ver = [](TensorListWrapper, TensorListWrapper, Arguments) -> void {
    ASSERT_FALSE(false);
  };

  this->RunTest(in, out, args, ver);
}


std::vector<Arguments> args1 = {{{"arg1", 1.}, {"arg2", 2.}, {"arg3", 3.}}};


INSTANTIATE_TEST_CASE_P(FirstOne, ExampleOperatorTestCase, ::testing::ValuesIn(args1));

TEST_P(ExampleOperatorTestCase, ExamplePTest1) {

  auto ver = [](TensorListWrapper, TensorListWrapper, Arguments) -> void {
    ASSERT_FALSE(false);
  };

  this->RunTest(in_, out_, GetParam(), ver);
}


INSTANTIATE_TEST_CASE_P(SecondOne, ExampleOperatorTestCase, ::testing::ValuesIn(args1));

TEST_P(ExampleOperatorTestCase, ExamplePTest2) {

  auto ver = [](TensorListWrapper, TensorListWrapper, Arguments) -> void {
    ASSERT_FALSE(false);
  };

  this->RunTest(in_, out_, GetParam(), ver);
}

} // namespace testing

} // namespace dali