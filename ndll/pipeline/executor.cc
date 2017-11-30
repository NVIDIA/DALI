#include "ndll/pipeline/executor.h"

namespace ndll {

void Executor::SetupDataForGraph(OpGraph *graph) {
  ClearMembers();

  // Create workspaces for each operator
  cpu_op_data_.resize(graph->NumCPUOp());
  internal_op_data_.resize(graph->NumInternalOp());
  gpu_op_data_.resize(graph->NumGPUOp());

  // Setup cpu op input and output buffers
  for (int i = 0; i < graph->NumCPUOp(); ++i) {
    CPUOpNode &node = graph->cpu_node(i);
    HostWorkspace &ws = cpu_op_data_[i];
    
    for (int j = 0; j < node.spec.NumInput(); ++j) {
      // Go get each set of input Tensors and add
      // them to this cpu ops workspace.
      NodeID parent_node_id = node.input_src_and_idx[j].first;
      NDLLOpType parent_op_type = graph->NodeType(parent_node_id);
      NDLL_ENFORCE(parent_op_type == NDLL_CPU,
          "Executor encountered cpu op with non-cpu input.");
      int parent_idx = graph->NodeIdx(parent_node_id);
      int input_src_idx = node.input_src_and_idx[j].second;

      HostWorkspace &src_ws = cpu_op_data_[parent_idx];
      const auto input = src_ws.Outputs<CPUBackend>(input_src_idx);
      ws.AddInput(input);
    }

    for (int j = 0; j < node.spec.NumOutput(); ++j) {
      // Allocate `batch_size` Tensors for this ops
      // results and add them to the workspace.
      vector<shared_ptr<Tensor<CPUBackend>>> output(batch_size_, nullptr);
      for (auto &tensor_ptr : output) tensor_ptr.reset(new Tensor<CPUBackend>);
      
      ws.AddOutput(output);
    }
  }

  // Setup internal op input and output buffers
  

  // Setup gpu op input and output buffers
  

  // Pre-size all outputs with the specified hint
  
  
  // Setup & allocation of mega-buffer
  // (if we move to fixed size mega-buffer)

  // Stream assignment in another function
}

void ThreadedExecutor::RunCPU() {

}

void ThreadedExecutor::RunInternal() {

}

void ThreadedExecutor::RunGPU() {

}

} // namespace ndll
