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
  ASSERT_EQ(graph.NumCPUOp(), 2);
  ASSERT_EQ(graph.NumInternalOp(), 0);
  ASSERT_EQ(graph.NumGPUOp(), 0);

  // Validate the source op
  auto node = graph.node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 1);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(node.children[0], 1);
  ASSERT_EQ(node.input_src_and_idx.size(), 0);
  
  // Validate copy op
  node = graph.node(1);
  ASSERT_EQ(node.id, 1);
  ASSERT_EQ(node.children.size(), 0);
  ASSERT_EQ(node.parents.size(), 1);
  ASSERT_EQ(node.parents[0], 0);
  ASSERT_EQ(node.input_src_and_idx.size(), 1);
  ASSERT_EQ(node.input_src_and_idx[0], std::make_pair(NodeID(0), 0));
}

TEST_F(OpGraphTest, TestGPUOnly) {
  OpGraph graph;

  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "gpu")
          .AddArg("inplace", true)
          .AddOutput("external_data", "gpu")
          ));
    
  graph.AddOp(this->PrepareSpec(
          OpSpec("CopyOp")
          .AddArg("device", "gpu")
          .AddInput("external_data", "gpu")
          .AddOutput("copy_data", "gpu")
          ));

  // Validate the graph
  ASSERT_EQ(graph.NumCPUOp(), 0);
  ASSERT_EQ(graph.NumInternalOp(), 0);
  ASSERT_EQ(graph.NumGPUOp(), 2);

  // Validate the source op
  auto node = graph.node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 1);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(node.children[0], 1);
  ASSERT_EQ(node.input_src_and_idx.size(), 0);
  
  // Validate copy op
  node = graph.node(1);
  ASSERT_EQ(node.id, 1);
  ASSERT_EQ(node.children.size(), 0);
  ASSERT_EQ(node.parents.size(), 1);
  ASSERT_EQ(node.parents[0], 0);
  ASSERT_EQ(node.input_src_and_idx.size(), 1);
  ASSERT_EQ(node.input_src_and_idx[0], std::make_pair(NodeID(0), 0));
}

TEST_F(OpGraphTest, TestCPUToGPU) {
  OpGraph graph;

  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "cpu")
          .AddArg("inplace", true)
          .AddOutput("external_data", "cpu")
          ));

  graph.AddOp(this->PrepareSpec(
          OpSpec("CopyToDevice")
          .AddArg("device", "internal")
          .AddInput("external_data", "cpu")
          .AddOutput("external_data", "gpu")
          ));
  
  graph.AddOp(this->PrepareSpec(
          OpSpec("CopyOp")
          .AddArg("device", "gpu")
          .AddInput("external_data", "gpu")
          .AddOutput("copy_data", "gpu")
          ));

  // Validate the graph
  ASSERT_EQ(graph.NumCPUOp(), 1);
  ASSERT_EQ(graph.NumInternalOp(), 1);
  ASSERT_EQ(graph.NumGPUOp(), 1);

  // Validate the source op
  auto node = graph.node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 1);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(node.children[0], 1);
  ASSERT_EQ(node.input_src_and_idx.size(), 0);
  
  // Validate copy-to-dev op
  node = graph.node(1);
  ASSERT_EQ(node.id, 1);
  ASSERT_EQ(node.children.size(), 1);
  ASSERT_EQ(node.parents.size(), 1);
  ASSERT_EQ(node.parents[0], 0);
  ASSERT_EQ(node.children[0], 2);
  ASSERT_EQ(node.input_src_and_idx.size(), 1);
  ASSERT_EQ(node.input_src_and_idx[0], std::make_pair(NodeID(0), 0));
  
  // Validate copy op
  node = graph.node(2);
  ASSERT_EQ(node.id, 2);
  ASSERT_EQ(node.children.size(), 0);
  ASSERT_EQ(node.parents.size(), 1);
  ASSERT_EQ(node.parents[0], 1);
  ASSERT_EQ(node.input_src_and_idx.size(), 1);
  ASSERT_EQ(node.input_src_and_idx[0], std::make_pair(NodeID(1), 0));
}

TEST_F(OpGraphTest, TestGPUThenCPUTopological) {
  OpGraph graph;

  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "gpu")
          .AddArg("inplace", true)
          .AddOutput("external_dev_data", "gpu")
          ));
  
  graph.AddOp(this->PrepareSpec(
          OpSpec("CopyOp")
          .AddArg("device", "gpu")
          .AddInput("external_dev_data", "gpu")
          .AddOutput("copy_data", "gpu")
          ));

  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "cpu")
          .AddArg("inplace", true)
          .AddOutput("external_host_data", "cpu")
          ));
    
  graph.AddOp(this->PrepareSpec(
          OpSpec("CopyOp")
          .AddArg("device", "cpu")
          .AddInput("external_host_data", "cpu")
          .AddOutput("copy_data", "cpu")
          ));

  // Validate the graph
  ASSERT_EQ(graph.NumCPUOp(), 2);
  ASSERT_EQ(graph.NumInternalOp(), 0);
  ASSERT_EQ(graph.NumGPUOp(), 2);

  // Validate the gpu source op
  auto node = graph.node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 1);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(node.children[0], 1);
  ASSERT_EQ(node.input_src_and_idx.size(), 0);
  
  // Validate gpu copy op
  node = graph.node(1);
  ASSERT_EQ(node.id, 1);
  ASSERT_EQ(node.children.size(), 0);
  ASSERT_EQ(node.parents.size(), 1);
  ASSERT_EQ(node.parents[0], 0);
  ASSERT_EQ(node.input_src_and_idx.size(), 1);
  ASSERT_EQ(node.input_src_and_idx[0], std::make_pair(NodeID(0), 0));
  
  // Validate cpu source op
  node = graph.node(2);
  ASSERT_EQ(node.id, 2);
  ASSERT_EQ(node.children.size(), 1);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(node.children[0], 3);
  ASSERT_EQ(node.input_src_and_idx.size(), 0);
  
  // Validate cpu copy op
  node = graph.node(3);
  ASSERT_EQ(node.id, 3);
  ASSERT_EQ(node.children.size(), 0);
  ASSERT_EQ(node.parents.size(), 1);
  ASSERT_EQ(node.parents[0], 2);
  ASSERT_EQ(node.input_src_and_idx.size(), 1);
  ASSERT_EQ(node.input_src_and_idx[0], std::make_pair(NodeID(2), 0));
}

TEST_F(OpGraphTest, TestFailureCPUOpGPUInput) {
  OpGraph graph;

  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "gpu")
          .AddArg("inplace", true)
          .AddOutput("external_data", "gpu")
          ));
  
  ASSERT_THROW(
      graph.AddOp(this->PrepareSpec(
              OpSpec("CopyOp")
              .AddArg("device", "cpu")
              .AddInput("external_data", "gpu")
              .AddOutput("copy_data", "cpu")
              )),
      std::runtime_error
      );
}

TEST_F(OpGraphTest, TestFailureCPUToGPUOp) {
  OpGraph graph;

  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "gpu")
          .AddArg("inplace", true)
          .AddOutput("external_data", "gpu")
          ));
  
  ASSERT_THROW(
      graph.AddOp(this->PrepareSpec(
              OpSpec("CopyOp")
              .AddArg("device", "cpu")
              .AddInput("external_data", "cpu")
              .AddOutput("copy_data", "cpu")
              )),
      std::runtime_error
      );
}

TEST_F(OpGraphTest, TestFailureNonTopological) {
  OpGraph graph;

  ASSERT_THROW(
      graph.AddOp(this->PrepareSpec(
              OpSpec("CopyOp")
              .AddArg("device", "cpu")
              .AddInput("external_data", "cpu")
              .AddOutput("copy_data", "cpu")
              )),
      std::runtime_error
      );

  // Note: Just to make it clear what this verifies...
  // graph.AddOp(this->PrepareSpec(
  //         OpSpec("ExternalSource")
  //         .AddArg("device", "cpu")
  //         .AddArg("inplace", true)
  //         .AddOutput("external_data", "cpu")
  //         ));
}

TEST_F(OpGraphTest, TestFailureCircularOp) {
  OpGraph graph;

  ASSERT_THROW(
      graph.AddOp(this->PrepareSpec(
              OpSpec("CopyOp")
              .AddArg("device", "cpu")
              .AddInput("data", "cpu")
              .AddOutput("data", "cpu")
              )),
      std::runtime_error
      );
}

 


} // namespace ndll
