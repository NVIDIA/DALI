#include "ndll/pipeline/executor.h"

#include "ndll/test/ndll_test.h"

namespace ndll {

class ExecutorTest : public NDLLTest {
public:
  inline OpSpec PrepareSpec(OpSpec spec) {
    spec.AddArg("batch_size", 1)
      .AddArg("num_threads", 1)
      .AddArg("cuda_stream", 0)
      .AddArg("pixels_per_image_hint", 0);
    return spec;
  }

  inline void PruneGraph(Executor *exe, OpGraph *graph,
      vector<string> output_names) {
    exe->PruneUnusedGraphNodes(graph, output_names);
  }
  
protected:
};

TEST_F(ExecutorTest, TestPruneBasicGraph) {
  Executor exe(1, 0, 0, 1);
  
  // Build a basic cpu->gpu graph
  OpGraph graph;
  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddOutput("data1", "cpu")
          .AddOutput("data2", "cpu")
          ));

  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddInput("data1", "cpu")
          .AddOutput("data3", "cpu")
          ));
  
  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddInput("data1", "cpu")
          .AddOutput("data4", "cpu")
          ));

  vector<string> outputs = {"data3_cpu"};
  this->PruneGraph(&exe, &graph, outputs);

  // Validate the graph - op 2 should
  // have been pruned as its outputs
  // are unused.
  ASSERT_EQ(graph.NumCPUOp(), 2);
  ASSERT_EQ(graph.NumInternalOp(), 0);
  ASSERT_EQ(graph.NumGPUOp(), 0);

  // Validate the source op
  auto node = graph.node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 1);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(node.children.count(1), 1);
  ASSERT_EQ(graph.TensorSourceID(node.spec.Output(0)), 0);
  ASSERT_EQ(graph.TensorIdxInSource(node.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node.spec.Output(0)));

  node = graph.node(1);
  ASSERT_EQ(node.id, 1);
  ASSERT_EQ(node.children.size(), 0);
  ASSERT_EQ(node.parents.size(), 1);
  ASSERT_EQ(node.parents.count(0), 1);
  ASSERT_EQ(graph.TensorSourceID(node.spec.Output(0)), 1);
  ASSERT_EQ(graph.TensorIdxInSource(node.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node.spec.Output(0)));
  ASSERT_EQ(node.spec.Output(0), "data3_cpu");
}

TEST_F(ExecutorTest, TestPruneMultiple) {
  Executor exe(1, 0, 0, 1);
  
  // Build a basic cpu->gpu graph
  OpGraph graph;
  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddOutput("data1", "cpu")
          .AddOutput("data2", "cpu")
          ));

  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddInput("data1", "cpu")
          .AddOutput("data3", "cpu")
          ));
  
  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddInput("data1", "cpu")
          .AddOutput("data4", "cpu")
          ));

  vector<string> outputs = {"data1_cpu"};
  this->PruneGraph(&exe, &graph, outputs);

  // Validate the graph - op 1&2 should
  // have been pruned
  ASSERT_EQ(graph.NumCPUOp(), 1);
  ASSERT_EQ(graph.NumInternalOp(), 0);
  ASSERT_EQ(graph.NumGPUOp(), 0);

  // Validate the source op
  auto node = graph.node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 0);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(graph.TensorSourceID(node.spec.Output(0)), 0);
  ASSERT_EQ(graph.TensorIdxInSource(node.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node.spec.Output(0)));
  ASSERT_EQ(node.spec.NumOutput(), 2);
  ASSERT_EQ(node.spec.Output(0), "data1_cpu");
  ASSERT_EQ(node.spec.Output(1), "data2_cpu");
}

TEST_F(ExecutorTest, TestPruneRecursive) {
  Executor exe(1, 0, 0, 1);
  
  // Build a basic cpu->gpu graph
  OpGraph graph;
  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddOutput("data1", "cpu")
          ));

  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddInput("data1", "cpu")
          .AddOutput("data2", "cpu")
          ));
  
  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddInput("data2", "cpu")
          .AddOutput("data3", "cpu")
          ));

  vector<string> outputs = {"data1_cpu"};
  this->PruneGraph(&exe, &graph, outputs);
  
  // Validate the graph - op 1&2 should
  // have been pruned
  ASSERT_EQ(graph.NumCPUOp(), 1);
  ASSERT_EQ(graph.NumInternalOp(), 0);
  ASSERT_EQ(graph.NumGPUOp(), 0);

  // Validate the source op
  auto node = graph.node(0);
  ASSERT_EQ(node.id, 0);
  ASSERT_EQ(node.children.size(), 0);
  ASSERT_EQ(node.parents.size(), 0);
  ASSERT_EQ(graph.TensorSourceID(node.spec.Output(0)), 0);
  ASSERT_EQ(graph.TensorIdxInSource(node.spec.Output(0)), 0);
  ASSERT_TRUE(graph.TensorIsType<CPUBackend>(node.spec.Output(0)));
  ASSERT_EQ(node.spec.NumOutput(), 1);
  ASSERT_EQ(node.spec.Output(0), "data1_cpu");
}

TEST_F(ExecutorTest, TestPruneWholeGraph) {
  Executor exe(1, 0, 0, 1);
  
  // Build a basic cpu->gpu graph
  OpGraph graph;
  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddOutput("data1", "cpu")
          ));

  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddInput("data1", "cpu")
          .AddOutput("data2", "cpu")
          ));
  
  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "cpu")
          .AddInput("data2", "cpu")
          .AddOutput("data3", "cpu")
          ));

  vector<string> outputs = {"data_that_does_not_exist"};
  ASSERT_THROW(this->PruneGraph(&exe, &graph, outputs),
      std::runtime_error);
}

TEST_F(ExecutorTest, TestBasicGraph) {
  Executor exe(1, 0, 0, 1);

  // Build a basic cpu->gpu graph
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

  vector<string> outputs = {"copy_data"};
  exe.Build(&graph, outputs);
}

} // namespace ndll
