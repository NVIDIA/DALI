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
      const auto input = src_ws.SharedOutput<CPUBackend>(input_src_idx);
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
  for (int i = 0; i < graph->NumInternalOp(); ++i) {
    InternalOpNode &node = graph->internal_node(i);
    internal::MixedWorkspace &ws = internal_op_data_[i];

    for (int j = 0; j < node.spec.NumInput(); ++j) {
      // Go get each set of input Tensors and add
      // them to this internal ops workspace.
      NodeID parent_node_id = node.input_src_and_idx[j].first;
      NDLLOpType parent_op_type = graph->NodeType(parent_node_id);
      NDLL_ENFORCE(parent_op_type == NDLL_CPU,
          "Executor encoutered internal op with non-cpu input.");
      int parent_idx = graph->NodeIdx(parent_node_id);
      int input_src_idx = node.input_src_and_idx[j].second;

      HostWorkspace &src_ws = cpu_op_data_[parent_idx];
      const auto input = src_ws.SharedOutput<CPUBackend>(input_src_idx);
      ws.AddInput(input);
    }

    for (int j = 0; j < node.spec.NumOutput(); ++j) {
      if (node.spec.OutputDevice(j) == "cpu") {
        // Allocate TensorLists for this ops outputs
        ws.AddOutput(std::make_shared<TensorList<CPUBackend>>());
      } else if (node.spec.OutputDevice(j) == "gpu") {
        ws.AddOutput(std::make_shared<TensorList<GPUBackend>>());
      } else {
        NDLL_FAIL("Executor encoutered internal op with non-gpu/cpu output.");
      }
    }
  }

  // Setup gpu op input and output buffers
  for (int i = 0; i < graph->NumGPUOp(); ++i) {
    GPUOpNode &node = graph->gpu_node(i);
    DeviceWorkspace &ws = gpu_op_data_[i];

    for (int j = 0; j < node.spec.NumInput(); ++j) {
      // Go get each set of input Tensors and add
      // them to this internal ops workspace.
      NodeID parent_node_id = node.input_src_and_idx[j].first;
      NDLLOpType parent_op_type = graph->NodeType(parent_node_id);
      int parent_idx = graph->NodeIdx(parent_node_id);
      int input_src_idx = node.input_src_and_idx[j].second;

      if (parent_op_type == NDLL_INTERNAL) {
        internal::MixedWorkspace &src_ws = internal_op_data_[parent_idx];
        if (node.spec.InputDevice(j) == "cpu") {
          const auto input = src_ws.SharedOutput<CPUBackend>(input_src_idx);
          ws.AddInput(input);          
        } else if (node.spec.InputDevice(j) == "gpu") {
          const auto input = src_ws.SharedOutput<GPUBackend>(input_src_idx);
          ws.AddInput(input);
        } else {
          NDLL_FAIL("Executor encoutered gpu op with non-cpu/gpu input.");
        }
      } else if (parent_op_type == NDLL_GPU) {
        DeviceWorkspace &src_ws = gpu_op_data_[parent_idx];
        if (node.spec.InputDevice(j) == "cpu") {
          // Note: This path should currently never occur, as we
          // do not allow gpu ops to produce cpu data outputs.
          const auto input = src_ws.SharedOutput<CPUBackend>(input_src_idx);
          ws.AddInput(input);          
        } else if (node.spec.InputDevice(j) == "gpu") {
          const auto input = src_ws.SharedOutput<GPUBackend>(input_src_idx);
          ws.AddInput(input);
        } else {
          NDLL_FAIL("Executor encoutered gpu op with non-cpu/gpu input.");
        }
      } else {
        NDLL_FAIL("Executor encoutered gpu op with non-internal/gpu input.");
      }
    }

    for (int j = 0; j < node.spec.NumOutput(); ++j) {
      // Allocate TensorLists for this ops output
      ws.AddOutput(std::make_shared<TensorList<GPUBackend>>());
    }
  }

  SetupMegaBufferForGraph(graph);
}

void Executor::PresizeData() {
  // Note: At some point our graph has source nodes that
  // only have outputs (data readers or external inputs).
  // Thus, the set of all outputs buffers in our workspaces
  // represents all the unique buffers in our graph.
  for (auto &ws : cpu_op_data_) {
    for (int i = 0; i < ws.NumOutput(); ++i) {
      NDLL_ENFORCE(ws.NumOutputAtIdx(i) == batch_size_, "Executor "
          "encountered cpu op workspace where the number of tensors "
          "is not equal to the batch size.");
      NDLL_ENFORCE(ws.OutputIsType<CPUBackend>(i), "Executor "
          "encountered cpu op with non-cpu output.");
      for (int j = 0; j < ws.NumOutputAtIdx(i); ++j) {
        Tensor<CPUBackend> *tensor = ws.Output<CPUBackend>(i, j);
        // We set the type of the tensor to uint8 temporarily
        tensor->mutable_data<uint8>();
        tensor->Resize({(Index)bytes_per_sample_hint_});
      }
    }
  }

  for (auto &ws : internal_op_data_) {
    for (int i = 0; i < ws.NumOutput(); ++i) {
      if (ws.OutputIsType<CPUBackend>(i)) {
        TensorList<CPUBackend> *tl = ws.Output<CPUBackend>(i);
        tl->mutable_data<uint8>();
        tl->Resize({{(Index)bytes_per_sample_hint_*batch_size_}});
      } else {
        TensorList<GPUBackend> *tl = ws.Output<GPUBackend>(i);
        tl->mutable_data<uint8>();
        tl->Resize({{(Index)bytes_per_sample_hint_*batch_size_}});
      }
    }
  }
  
  for (auto &ws : gpu_op_data_) {
    for (int i = 0; i < ws.NumOutput(); ++i) {
      NDLL_ENFORCE(ws.OutputIsType<GPUBackend>(i), "Executor "
          "encountered gpu op with non-gpu output.");
      TensorList<GPUBackend> *tl = ws.Output<GPUBackend>(i);
      tl->mutable_data<uint8>();
      tl->Resize({{(Index)bytes_per_sample_hint_*batch_size_}});
    }
  }
}

void Executor::SetupMegaBufferForGraph(OpGraph *graph) {
  mega_buffer_.mutable_data<uint8>();
  mega_buffer_gpu_.mutable_data<uint8>();

  size_t total_bytes = 0;
  vector<size_t> offsets;
  vector<int> num_buffer_for_op(graph->NumGPUOp(), 0);
  for (int i = 0; i < graph->NumGPUOp(); ++i) {
    const vector<size_t> &sizes = graph->gpu_op(i)->KernelParameterSizes();
    num_buffer_for_op[i] = sizes.size();
    for (auto &num_bytes : sizes) {
      // Align the start of each buffer to 8-bytes
      size_t aligned_num_bytes = round_up_to_8(num_bytes);
      offsets.push_back(total_bytes);
      total_bytes += aligned_num_bytes;
    }
  }
  offsets.push_back(total_bytes);
  
  mega_buffer_.Resize({(Index)total_bytes});
  mega_buffer_gpu_.Resize({(Index)total_bytes});
  uint8 *mega_buffer_ptr = mega_buffer_.template mutable_data<uint8>();
  uint8 *mega_buffer_gpu_ptr = mega_buffer_gpu_.template mutable_data<uint8>();

  // Hand out buffers for all the ops kernel parameters
  int buffer_id = 0;
  for (int i = 0; i < graph->NumGPUOp(); ++i) {
    DeviceWorkspace &ws = gpu_op_data_[i];
    for (int j = 0; j < num_buffer_for_op[i]; ++j) {
      auto cpu_tensor = std::make_shared<Tensor<CPUBackend>>();
      auto gpu_tensor = std::make_shared<Tensor<GPUBackend>>();

      size_t offset = offsets[buffer_id];
      size_t buffer_size = offsets[buffer_id+1] - offsets[buffer_id];
      cpu_tensor->ShareData(mega_buffer_ptr + offset, buffer_size);      
      gpu_tensor->ShareData(mega_buffer_gpu_ptr + offset, buffer_size);
      
      ws.AddParamTensor(cpu_tensor, gpu_tensor);
      ++buffer_id;
    }
  }    
}

void ThreadedExecutor::RunCPU() {

}

void ThreadedExecutor::RunInternal() {

}

void ThreadedExecutor::RunGPU() {

}

} // namespace ndll
