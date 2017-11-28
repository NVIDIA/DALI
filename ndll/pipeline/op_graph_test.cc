#include "ndll/pipeline/op_graph.h"

#include <gtest/gtest.h>

#include "ndll/pipeline/operators/copy_op.h"
#include "ndll/test/ndll_test.h"

namespace ndll {

class OpGraphTest : public NDLLTest {
public:
  inline OpSpec PrepareSpec(OpSpec spec) {
    spec.AddArg("batch_size", 1)
      .AddArg("num_threads", 1)
      .AddArg("cuda_stream", 0)
      .AddArg("pixels_per_image_hint", 0);
    return spec;
  }
protected:
};

TEST_F(OpGraphTest, TestCPUOnly) {
  OpGraph graph;

  // Add copy op insertion
  // Add contiguous-ify op
  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "cpu")
          .AddArg("inplace", true)
          .AddOutput("external_data", "cpu")
          ));
  
  graph.AddOp(this->PrepareSpec(
          OpSpec("CopyOp")
          .AddInput("external_data", "cpu")
          .AddOutput("copy_data", "cpu")
          ));

  // Validate the graph
  ASSERT_EQ(graph.NumOpWithBackend<CPUBackend>(), 2);
  ASSERT_EQ(graph.NumOpWithBackend<GPUBackend>(), 0);

  // Validate the source op
  auto node = graph.node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 1);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(node.children[0], 1);

  // Validate copy op
  node = graph.node(1);
  ASSERT_EQ(node.id, 1);
  ASSERT_EQ(node.children.size(), 0);
  ASSERT_EQ(node.parents.size(), 1);
  ASSERT_EQ(node.parents[0], 0);
}

TEST_F(OpGraphTest, TestGPUOnly) {
 
}

TEST_F(OpGraphTest, TestPipeline) {

}

TEST_F(OpGraphTest, TestSplit) {

}

TEST_F(OpGraphTest, TestDiamond) {

}

TEST_F(OpGraphTest, TestTopologicalCheck) {

}

TEST_F(OpGraphTest, TestCPUToGPUAndBack) {

}

 


} // namespace ndll
