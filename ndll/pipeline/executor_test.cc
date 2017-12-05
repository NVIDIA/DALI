#include "ndll/pipeline/executor.h"

#include "ndll/test/ndll_test.h"

namespace ndll {

class ExecutorTest : public NDLLTest {
public:
  void SetUp() override {
    NDLLTest::SetUp();
    batch_size_ = jpeg_names_.size();
  }
  
  inline OpSpec PrepareSpec(OpSpec spec) {
    spec.AddArg("batch_size", batch_size_)
      .AddArg("num_threads", num_threads_)
      .AddArg("cuda_stream", 0)
      .AddArg("pixels_per_image_hint", 0);
    return spec;
  }

  inline void PruneGraph(Executor *exe, OpGraph *graph,
      vector<string> output_names) {
    exe->PruneUnusedGraphNodes(graph, output_names);
  }

  vector<HostWorkspace> CPUData(Executor *exe) {
    return exe->cpu_op_data_;
  }

  vector<internal::MixedWorkspace> InternalData(Executor *exe) {
    return exe->internal_op_data_;
  }

  vector<DeviceWorkspace> GPUData(Executor *exe) {
    return exe->gpu_op_data_;
  }
  
protected:
  int batch_size_, num_threads_ = 1;
};

TEST_F(ExecutorTest, TestPruneBasicGraph) {
  Executor exe(this->batch_size_, this->num_threads_, 0, 1);
  
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
  Executor exe(this->batch_size_, this->num_threads_, 0, 1);
  
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
  Executor exe(this->batch_size_, this->num_threads_, 0, 1);
  
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
  Executor exe(this->batch_size_, this->num_threads_, 0, 1);
  
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

TEST_F(ExecutorTest, TestSetupData) {
  Executor exe(this->batch_size_, this->num_threads_, 0, 1);

  // Build a basic cpu->gpu graph
  OpGraph graph;
  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "cpu")
          .AddArg("inplace", true)
          .AddOutput("external_data", "cpu")
          ));

  graph.AddOp(this->PrepareSpec(
          OpSpec("MakeContiguous")
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

  
}

TEST_F(ExecutorTest, TestDataSetup) {
  Executor exe(this->batch_size_, this->num_threads_, 0, 1);

  // Build a basic cpu->gpu graph
  OpGraph graph;
  graph.AddOp(this->PrepareSpec(
          OpSpec("ExternalSource")
          .AddArg("device", "cpu")
          .AddOutput("data1", "cpu")
          ));

  graph.AddOp(this->PrepareSpec(
          OpSpec("MakeContiguous")
          .AddArg("device", "internal")
          .AddInput("data1", "cpu")
          .AddOutput("data2", "gpu")
          ));
  
  graph.AddOp(this->PrepareSpec(
          OpSpec("DummyOp")
          .AddArg("device", "gpu")
          .AddInput("data2", "gpu")
          .AddOutput("data3", "gpu")
          ));

  vector<string> outputs = {"data3_gpu"};
  exe.Build(&graph, outputs);

  // Verify the data has been setup correctly
  auto host_workspaces = this->CPUData(&exe);
  ASSERT_EQ(host_workspaces.size(), 1);
  HostWorkspace &hws = host_workspaces[0];
  ASSERT_EQ(hws.NumInput(), 0);
  ASSERT_EQ(hws.NumOutput(), 1);
  ASSERT_EQ(hws.NumOutputAtIdx(0), batch_size_);
  ASSERT_TRUE(hws.OutputIsType<CPUBackend>(0));

  auto internal_workspaces = this->InternalData(&exe);
  ASSERT_EQ(internal_workspaces.size(), 1);
  internal::MixedWorkspace &mws = internal_workspaces[0];
  ASSERT_EQ(mws.NumInput(), 1);
  ASSERT_EQ(mws.NumInputAtIdx(0), batch_size_);
  ASSERT_TRUE(mws.InputIsType<CPUBackend>(0));
  ASSERT_EQ(mws.NumOutput(), 1);
  ASSERT_TRUE(mws.OutputIsType<GPUBackend>(0));

  auto device_workspaces = this->GPUData(&exe);
  ASSERT_EQ(device_workspaces.size(), 1);
  DeviceWorkspace &dws = device_workspaces[0];
  ASSERT_EQ(dws.NumInput(), 1);
  ASSERT_TRUE(dws.InputIsType<GPUBackend>(0));
  ASSERT_EQ(dws.NumOutput(), 1);
  ASSERT_TRUE(dws.OutputIsType<GPUBackend>(0));
}

} // namespace ndll
