#include <gtest/gtest.h>
#include <dali/test/op_graph.h>

namespace dali {
namespace testing {

TEST(OpGraph, ValidDAG1) {

  OpDAG dag;
  auto n1 = dag.add("MyOp1");
  auto n2 = dag.add("MyOp2");
  auto n3 = dag.add("MyOp3");
  dag.output << n3(0);
  n3 << n1(0), n2(0);
  n1 << dag.input(0);
  n2 << dag.input(1);
  EXPECT_TRUE(dag.validate(true));
}

TEST(OpGraph, ValidDAG2) {

  OpDAG dag;
  auto n1 = dag.add("MyOp1");
  auto n2 = dag.add("MyOp2");
  auto n3 = dag.add("MyOp3");
  dag.output << n3(0);
  n3 << n1(0), n2(0);
  n1 << dag.input(0);
  n2 << dag.input(1), n1(0);
  EXPECT_TRUE(dag.validate(true));
}

TEST(OpGraph, InvalidDAG) {

  OpDAG dag;
  auto n1 = dag.add("MyOp1");
  auto n2 = dag.add("MyOp2");
  auto n3 = dag.add("MyOp3");
  dag.output << n3(0);
  n3 << n1(0), n2(0);
  n1 << dag.input(0);
  n2 << dag.input(1), n3(0);
  EXPECT_FALSE(dag.validate(true));
}

}  // testing
}  // namespace dali
