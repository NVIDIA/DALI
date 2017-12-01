#include "ndll/pipeline/executor.h"

#include <unordered_map>
#include <unordered_set>

namespace ndll {

void Executor::SetupDataForGraph(OpGraph *graph) {
  // Clear any old data setup
  cpu_op_data_.clear();
  internal_op_data_.clear();
  gpu_op_data_.clear();

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

void Executor::SetupStreamsForGraph(OpGraph *graph) {
  // Note: Basic stream assignment algorithm -
  // Each node can reuse its parents stream as
  // long as no other node in the same front
  // has already claimed it. Each op is associated
  // with an event, which is inserted into its
  // stream after it is executed. Its child ops
  // that are not in the same stream will make
  // their stream block on the parent event prior
  // to executing anything in their stream.
  gpu_op_events_.clear();
  gpu_op_parent_events_.clear();

  // We will traverse the graph breadth-first,
  // initialize queue with the ids of our gpu
  // root nodes.
  std::unordered_set<NodeID> processed_nodes;
  std::queue<NodeID> node_queue;
  for (int i = 0; i < graph->NumGPUOp(); ++i) {
    // An op is a root if it has no gpu inputs
    GPUOpNode &node = graph->gpu_node(i);
    
    bool is_root = true;
    for (auto &parent_id : node.parents) {
      if (graph->NodeType(parent_id) == NDLL_GPU) {
        is_root = false;
        break;
      }
    }

    if (is_root) {
      node_queue.push(node.id);
      NDLL_ENFORCE(processed_nodes.insert(node.id).second);
    }

    // Create an event that will signal the
    // completion of this ops computation
    gpu_op_events_.push_back(event_pool_.GetEvent());
  }

  std::unordered_map<NodeID, cudaStream_t> node_streams_;
  while (!node_queue.empty()) {
    const OpNode &current_node = graph->node(node_queue.front());
    int current_op_idx = graph->NodeIdx(current_node.id);
    DeviceWorkspace &ws = gpu_op_data_[current_op_idx];
    node_queue.pop();

    bool found_stream = false;
    for (const auto &parent_id : current_node.parents) {
      // We only care about gpu parent ops
      if (graph->NodeType(parent_id) == NDLL_INTERNAL) continue;
      
      // If a parent stream is still available, take it
      auto it = node_streams_.find(parent_id);
      if (it != node_streams_.end()) {
        // Add the stream to this ops workspace
        cudaStream_t stream = it->second;
        ws.set_stream(stream);

        // Remove this stream from the parents entry
        // and add it to the entry for this op
        node_streams_.erase(it);
        NDLL_ENFORCE(node_streams_.insert({current_node.id, stream}).second);

        found_stream = true;
        break;
      }

      // If we don't use this parent's stream, we'll need to block
      // on its event to ensure the dependency is respected.
      int parent_op_idx = graph->NodeIdx(parent_id);
      cudaEvent_t parent_event = gpu_op_events_[parent_op_idx];
      gpu_op_parent_events_[current_op_idx].push_back(parent_event);
    }

    if (!found_stream) {
      // Couldn't reuse any parent streams, request
      // a new stream from the stream pool
      cudaStream_t stream = stream_pool_.GetStream();
      ws.set_stream(stream);
      NDLL_ENFORCE(node_streams_.insert({current_node.id, stream}).second,
          "Internal error, GPU op stream insertion failed.");
    }

    // Now add this ops un-processed children to the queue
    for (const auto &child_id : current_node.children) {
      auto it = processed_nodes.find(child_id);
      if (it != processed_nodes.end()) {
        node_queue.push(child_id);
        NDLL_ENFORCE(processed_nodes.insert(child_id).second);
      }
    }
  }
}

} // namespace ndll
