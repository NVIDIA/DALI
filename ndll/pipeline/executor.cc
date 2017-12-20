#include "ndll/pipeline/executor.h"

#include <algorithm>
#include <iterator>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace ndll {

void Executor::Build(OpGraph *graph, vector<string> output_names) {
  NDLL_ENFORCE(graph != nullptr, "Input graph is nullptr.");
  NDLL_ENFORCE(graph->NumOp() > 0, "Graph has no operators.");
  output_names_ = output_names;
  graph_ = graph;
  
  // Remove any node from the graph whose output
  // will not be used as an output or by another node
  PruneUnusedGraphNodes();

  // Setup workspaces for each op and connect
  // their inputs and outputs.
  WorkspaceBlob base_wsb;
  SetupDataForGraph(&base_wsb);

  // Presize the workspaces based on the hint
  PresizeData(&base_wsb);
  
  // Assign streams to all internal & gpu ops
  SetupStreamsForGraph(&base_wsb);

  SetupOutputQueuesForGraph();
  
  // For each set of outputs, setup another set of
  // workspaces so that nothing has to be altered
  // during execution (this is necessary for
  // asynchonrous executors that can overlap work issue)
  for (int i = 0; i < queue_depth_; ++i) {
    SetOutputBuffersForIter(i, &base_wsb);
    wss_.push_back(base_wsb);
  }
}

void Executor::RunCPU() {
  // Block until there is a free buffer to use
  std::unique_lock<std::mutex> lock(free_mutex_);
  while (free_queue_.empty()) {
    free_cond_.wait(lock);
  }
  int queue_idx = free_queue_.front();
  free_queue_.pop();
  lock.unlock();
  
  // Run the cpu-ops in the thread pool
  WorkspaceBlob &wsb = wss_[queue_idx];
  for (int i = 0; i < batch_size_; ++i) {
    thread_pool_.DoWorkWithID(std::bind(
            [this, &wsb] (int data_idx, int tid) {
              SampleWorkspace ws;
              for (int j = 0; j < graph_->NumCPUOp(); ++j) {
                Operator<CPUBackend> &op = graph_->cpu_op(j);
                wsb.cpu_op_data[j].GetSample(&ws, data_idx, tid);
                op.Run(&ws);
              }
            }, i, std::placeholders::_1));
  }
  thread_pool_.WaitForWork();

  // Pass the work to the internal stage
  std::unique_lock<std::mutex> internal_lock(internal_mutex_);
  internal_work_queue_.push(queue_idx);
  internal_lock.unlock();
}

void Executor::RunInternal() {
  std::unique_lock<std::mutex> lock(internal_mutex_);
  NDLL_ENFORCE(!internal_work_queue_.empty(), "Internal work "
      "queue empty. Did you call RunCPU prior to RunInternal?");
  int queue_idx = internal_work_queue_.front();
  internal_work_queue_.pop();
  lock.unlock();

  WorkspaceBlob &wsb = wss_[queue_idx];
  for (int i = 0; i < graph_->NumInternalOp(); ++i) {
    internal::InternalOp &op = graph_->internal_op(i);
    internal::MixedWorkspace &ws = wsb.internal_op_data[i];
    op.Run(&ws);
    if (ws.has_stream() && ws.has_event()) {
      CUDA_CALL(cudaEventRecord(ws.event(), ws.stream()));
    }
  }

  // Pass the work to the gpu stage
  std::unique_lock<std::mutex> gpu_lock(gpu_mutex_);
  gpu_work_queue_.push(queue_idx);
  gpu_lock.unlock();
}

void Executor::RunGPU() {
  std::unique_lock<std::mutex> gpu_lock(gpu_mutex_);
  NDLL_ENFORCE(!gpu_work_queue_.empty(), "GPU work queue "
      "empty. Did you call RunInternal prior to RunGPU?");
  int queue_idx = gpu_work_queue_.front();
  gpu_work_queue_.pop();
  gpu_lock.unlock();
  
  // Enforce our assumed dependency between consecutive
  // iterations of a stage of the pipeline.
  if (previous_gpu_queue_idx_ != -1) {
    for (size_t i = 0; i < output_names_.size(); ++i) {
      if (graph_->TensorIsType<CPUBackend>(output_names_[i])) continue;
      CUDA_CALL(cudaEventSynchronize(
              gpu_output_events_[i].GetEvent(previous_gpu_queue_idx_)));
    }
  }

  WorkspaceBlob &wsb = wss_[queue_idx];
  for (int i = 0; i < graph_->NumGPUOp(); ++i) {
    Operator<GPUBackend> &op = graph_->gpu_op(i);
    DeviceWorkspace &ws = wsb.gpu_op_data[i];
    auto parent_events = ws.ParentEvents();

    for (auto &event : parent_events) {
      CUDA_CALL(cudaStreamWaitEvent(ws.stream(), event, 0));
    }
    op.Run(&ws);
    if (ws.has_event()) {
      CUDA_CALL(cudaEventRecord(ws.event(), ws.stream()));
    }
  }

  for (size_t i = 0; i < output_names_.size(); ++i) {
    if (graph_->TensorIsType<CPUBackend>(output_names_[i])) continue;
    NodeID src_id = graph_->TensorSourceID(output_names_[i]);
    int src_idx = graph_->NodeIdx(src_id);

    // Record events for each output requested by the user
    cudaEvent_t event = gpu_output_events_[i].GetEvent(queue_idx);
    if (graph_->NodeType(src_id) == NDLL_INTERNAL) {
      auto &ws = wsb.internal_op_data[src_idx];
      CUDA_CALL(cudaEventRecord(event, ws.stream()));   
    } else if (graph_->NodeType(src_id) == NDLL_GPU) {
      auto &ws = wsb.gpu_op_data[src_idx];
      CUDA_CALL(cudaEventRecord(event, ws.stream()));   
    } else {
      NDLL_FAIL("Internal error. Output node is not gpu/internal");
    }
  }

  // Update the ready queue to signal that all the work
  // in the `queue_idx` set of output buffers has been
  // issued. Notify any waiting threads.
  std::unique_lock<std::mutex> lock(ready_mutex_);
  ready_queue_.push(queue_idx);
  ready_cond_.notify_one();
  lock.unlock();

  // Save the queue_idx so we can enforce the
  // dependency between consecutive iterations
  // of the gpu stage of the pipeline.
  previous_gpu_queue_idx_ = queue_idx;
}

void Executor::Outputs(DeviceWorkspace *ws) {
  NDLL_ENFORCE(ws != nullptr, "Workspace is nullptr");
  ws->Clear();

  // Mark the last in-use buffer as free and signal
  // to waiting threads
  if (!in_use_queue_.empty()) {
    std::unique_lock<std::mutex> lock(free_mutex_);
    free_queue_.push(in_use_queue_.front());
    in_use_queue_.pop();
    free_cond_.notify_one();
    lock.unlock();
  }
  
  // Block until the work for a batch has been issued.
  // Move the queue id from ready to in_use
  std::unique_lock<std::mutex> lock(ready_mutex_);
  while (ready_queue_.empty()) {
    ready_cond_.wait(lock);
  }
  int output_idx = ready_queue_.front();
  ready_queue_.pop();
  in_use_queue_.push(output_idx);
  lock.unlock();

  // Gather the results TensorLists and block on their
  // events to make sure that the computation has completed
  for (size_t i = 0; i < output_names_.size(); ++i) {
    auto it = type_idx_map_.find(output_names_[i]);
    NDLL_ENFORCE(it != type_idx_map_.end(), "Executor could not "
        "find output with name '" + output_names_[i] + "'.");

    if (graph_->TensorIsType<CPUBackend>(output_names_[i])) {
      auto &tl_pool = cpu_outputs_[it->second];
      ws->AddOutput(tl_pool.GetTL(output_idx));
    } else {
      auto &tl_pool = gpu_outputs_[it->second];
      ws->AddOutput(tl_pool.GetTL(output_idx));
      CUDA_CALL(cudaEventSynchronize(
              gpu_output_events_[i].GetEvent(output_idx)));
    }
  }
}

void Executor::PruneUnusedGraphNodes() {
  // We want to remove any nodes whose outputs are
  // never used by another node or as an output
  NDLL_ENFORCE(output_names_.size() > 0,
      "No outputs requested, nothing to execute.");

  while (true) {
    // We do not edit the graph while we are iterating
    // as node ids will be updated when an op is removed
    vector<NodeID> to_remove;
    for (int i = 0; i < graph_->NumOp(); ++i) {
      OpNode &node = graph_->node(i);
      
      // If this node has children, don't prune it
      if (!node.children.empty()) continue;

      // Note: this is technically a very innefficient
      // way to find the insertsection of the node outputs
      // and the outputs of the graph. The number of outputs
      // is usually 1-2, so it shouldn't matter
      bool found_match = false;
      for (int j = 0; j < node.spec.NumOutput(); ++j) {
        for (size_t k = 0; k < output_names_.size(); ++k) {
          if (node.spec.Output(j) == output_names_[k]) {
            found_match = true;
            break;
          }
        }
        if (found_match) break;
      }

      // If this node produces an output, don't prune it
      if (found_match) continue;
      
      // Mark the node for pruning
      to_remove.push_back(node.id);
    }

    // No nodes were removed, pruning complete
    if (to_remove.size() == 0) break;
    
    for (size_t i = 0; i < to_remove.size(); ++i) {
      // Note: After deleting a node, the graph updates
      // all other nodes in the graph to keep the node
      // ids consisten with the number of nodes in the
      // graph. 'to_remove' will store the removal
      // targets largest to smallest, so we just subtract
      // the number of previously deleted nodes from
      // the current node id.
      graph_->RemoveOp(to_remove[i] - i);
    }
  }

  // If we've pruned the entire graph, something has gone wrong
  NDLL_ENFORCE(graph_->NumOp() > 0, "No output names match "
      "data produced by the pipeline.");
}

void Executor::SetupDataForGraph(WorkspaceBlob *wsb) {
  // Clear any old data setup
  wsb->cpu_op_data.clear();
  wsb->internal_op_data.clear();
  wsb->gpu_op_data.clear();

  // Create workspaces for each operator
  wsb->cpu_op_data.resize(graph_->NumCPUOp());
  wsb->internal_op_data.resize(graph_->NumInternalOp());
  wsb->gpu_op_data.resize(graph_->NumGPUOp());

  // Setup cpu op input and output buffers
  for (int i = 0; i < graph_->NumCPUOp(); ++i) {
    CPUOpNode &node = graph_->cpu_node(i);
    HostWorkspace &ws = wsb->cpu_op_data[i];
    
    for (int j = 0; j < node.spec.NumInput(); ++j) {
      // Go get each set of input Tensors and add
      // them to this cpu ops workspace.
      NodeID parent_node_id = graph_->TensorSourceID(node.spec.Input(j));
      NDLLOpType parent_op_type = graph_->NodeType(parent_node_id);
      NDLL_ENFORCE(parent_op_type == NDLL_CPU,
          "Executor encountered cpu op with non-cpu input.");
      int parent_idx = graph_->NodeIdx(parent_node_id);
      int input_src_idx = graph_->TensorIdxInSource(node.spec.Input(j));

      HostWorkspace &src_ws = wsb->cpu_op_data[parent_idx];
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
  for (int i = 0; i < graph_->NumInternalOp(); ++i) {
    InternalOpNode &node = graph_->internal_node(i);
    internal::MixedWorkspace &ws = wsb->internal_op_data[i];

    for (int j = 0; j < node.spec.NumInput(); ++j) {
      // Go get each set of input Tensors and add
      // them to this internal ops workspace.
      NodeID parent_node_id = graph_->TensorSourceID(node.spec.Input(j));
      NDLLOpType parent_op_type = graph_->NodeType(parent_node_id);
      NDLL_ENFORCE(parent_op_type == NDLL_CPU,
          "Executor encoutered internal op with non-cpu input.");
      int parent_idx = graph_->NodeIdx(parent_node_id);
      int input_src_idx = graph_->TensorIdxInSource(node.spec.Input(j));

      HostWorkspace &src_ws = wsb->cpu_op_data[parent_idx];
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
  for (int i = 0; i < graph_->NumGPUOp(); ++i) {
    GPUOpNode &node = graph_->gpu_node(i);
    DeviceWorkspace &ws = wsb->gpu_op_data[i];
    
    for (int j = 0; j < node.spec.NumInput(); ++j) {
      // Go get each set of input Tensors and add
      // them to this internal ops workspace.
      NodeID parent_node_id = graph_->TensorSourceID(node.spec.Input(j));
      NDLLOpType parent_op_type = graph_->NodeType(parent_node_id);
      int parent_idx = graph_->NodeIdx(parent_node_id);
      int input_src_idx = graph_->TensorIdxInSource(node.spec.Input(j));

      if (parent_op_type == NDLL_INTERNAL) {
        internal::MixedWorkspace &src_ws = wsb->internal_op_data[parent_idx];
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
        DeviceWorkspace &src_ws = wsb->gpu_op_data[parent_idx];
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
}

void Executor::PresizeData(WorkspaceBlob *wsb) {
  // Note: At some point our graph has source nodes that
  // only have outputs (data readers or external inputs).
  // Thus, the set of all outputs buffers in our workspaces
  // represents all the unique buffers in our graph.
  for (auto &ws : wsb->cpu_op_data) {
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

  for (auto &ws : wsb->internal_op_data) {
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
  
  for (auto &ws : wsb->gpu_op_data) {
    for (int i = 0; i < ws.NumOutput(); ++i) {
      NDLL_ENFORCE(ws.OutputIsType<GPUBackend>(i), "Executor "
          "encountered gpu op with non-gpu output.");
      TensorList<GPUBackend> *tl = ws.Output<GPUBackend>(i);
      tl->mutable_data<uint8>();
      tl->Resize({{(Index)bytes_per_sample_hint_*batch_size_}});
    }
  }
}

void Executor::SetupStreamsForGraph(WorkspaceBlob *wsb) {
  for (int i = 0; i < graph_->NumInternalOp(); ++i) {
    // For internal ops, we assign unique streams to each
    // op. This ensures (assuming the stream pool does not
    // have a limit) that we won't have false dependencies
    // between internal ops and the previous iterations
    // gpu ops.
    internal::MixedWorkspace &ws = wsb->internal_op_data[i];
    ws.set_stream(stream_pool_.GetStream());
    ws.set_event(event_pool_.GetEvent());
  }
  
  // Note: Basic stream assignment algorithm -
  // Each node can reuse its parents stream as
  // long as no other node in the same front
  // has already claimed it. Each op is associated
  // with an event, which is inserted into its
  // stream after it is executed. Its child ops
  // that are not in the same stream will make
  // their stream block on the parent event prior
  // to executing anything in their stream.

  // We will traverse the graph breadth-first,
  // initialize queue with the ids of our gpu
  // root nodes.
  std::unordered_set<NodeID> processed_nodes;
  std::queue<NodeID> node_queue;
  for (int i = 0; i < graph_->NumGPUOp(); ++i) {
    // An op is a root if it has no gpu inputs
    GPUOpNode &node = graph_->gpu_node(i);
    
    bool is_root = true;
    for (auto &parent_id : node.parents) {
      if (graph_->NodeType(parent_id) == NDLL_GPU) {
        is_root = false;
        break;
      }
    }

    // If this op has more than a single child node,
    // we will need an event to synchronize the child
    // that does not get the parents stream
    //
    // TODO(tgale): We could do this more efficiently by
    // applying the same algorithm we use to assign streams
    if (node.children.size() > 1) {
      DeviceWorkspace &ws = wsb->gpu_op_data[graph_->NodeIdx(i)];
      ws.set_event(event_pool_.GetEvent());
    }
    
    if (is_root) {
      node_queue.push(node.id);
      NDLL_ENFORCE(processed_nodes.insert(node.id).second);
    }
  }

  std::unordered_map<NodeID, cudaStream_t> node_streams;
  while (!node_queue.empty()) {
    const OpNode &current_node = graph_->node(node_queue.front());
    int current_op_idx = graph_->NodeIdx(current_node.id);
    DeviceWorkspace &ws = wsb->gpu_op_data[current_op_idx];
    node_queue.pop();

    bool found_stream = false;
    auto it = current_node.parents.begin();
    for (; it != current_node.parents.end(); ++it) {
      NodeID parent_id = *it;
      int parent_op_idx = graph_->NodeIdx(parent_id);

      if (graph_->NodeType(parent_id) == NDLL_INTERNAL) {
        // We will not re-use internal op streams, but
        // we will need to block on this ops event to
        // make sure that we respect the dependency
        internal::MixedWorkspace parent_ws = wsb->internal_op_data[parent_op_idx];
        ws.AddParentEvent(parent_ws.event());
      } else if (graph_->NodeType(parent_id) == NDLL_GPU) {
        // If a parent stream is still available, take it
        auto it = node_streams.find(parent_id);
        if (it != node_streams.end()) {
          // Add the stream to this ops workspace
          cudaStream_t stream = it->second;
          ws.set_stream(stream);
          
          // Remove this stream from the parents entry
          // and add it to the entry for this op
          node_streams.erase(it);
          NDLL_ENFORCE(node_streams.insert({current_node.id, stream}).second);
          
          found_stream = true;
          break;
        }

        // If we don't use this parent's stream, we'll need to block
        // on its event to ensure the dependency is respected.
        DeviceWorkspace parent_ws = wsb->gpu_op_data[parent_op_idx];
        ws.AddParentEvent(parent_ws.event());
      } else {
        NDLL_FAIL("Executor encountered gpu op with non-gpu/internal parent.");
      }
    }

    // Note: We want to finish iterating over the parents,
    // but we do not want to repeat a node. If the earlier
    // loop terminated because it reached the end, do not
    // increment the iterator to the next parent or it will
    // be out of the valid range
    if (it != current_node.parents.end()) ++it;
    
    // Make sure we finish adding all parents events
    // if we exited the prevous loop early
    for (; it != current_node.parents.end(); ++it) {
      NodeID parent_id = *it;
      int parent_op_idx = graph_->NodeIdx(parent_id);
      
      if (graph_->NodeType(parent_id) == NDLL_INTERNAL) {
        internal::MixedWorkspace parent_ws = wsb->internal_op_data[parent_op_idx];
        ws.AddParentEvent(parent_ws.event());
      } else if (graph_->NodeType(parent_id) == NDLL_GPU) {
        DeviceWorkspace parent_ws = wsb->gpu_op_data[parent_op_idx];
        ws.AddParentEvent(parent_ws.event());
      } else {
        NDLL_FAIL("Executor encountered gpu op with non-gpu/internal parent.");
      }
    }
    
    if (!found_stream) {
      // Couldn't reuse any parent streams, request
      // a new stream from the stream pool
      cudaStream_t stream = stream_pool_.GetStream();
      ws.set_stream(stream);
      NDLL_ENFORCE(node_streams.insert({current_node.id, stream}).second,
          "Internal error, GPU op stream insertion failed.");
    }

    // Now add this ops un-processed children to the queue
    for (const auto &child_id : current_node.children) {
      auto it = processed_nodes.find(child_id);
      if (it == processed_nodes.end()) {
        node_queue.push(child_id);
        NDLL_ENFORCE(processed_nodes.insert(child_id).second);
      }
    }
  }
}

void Executor::SetupOutputQueuesForGraph() {
  // Allocate output TensorList pools for each output
  for (auto &name : output_names_) {
    auto tensor_meta = graph_->TensorSourceMeta(name);

    // Collect meta-data about the tensor for fast lookup later.
    OutputInfo info;
    info.prod_and_idx = std::make_pair(tensor_meta.node, tensor_meta.index);
    vector<TensorMeta> consumer_meta = graph_->TensorConsumerMeta(name);
    for (auto &meta : consumer_meta) {
      auto tmp = std::make_pair(meta.node, meta.index);
      info.con_and_idx.push_back(tmp);
    }

    // Create the buffer and events
    if (tensor_meta.is_cpu) {
      cpu_outputs_.push_back(TensorListPool<CPUBackend>(
              queue_depth_, batch_size_, bytes_per_sample_hint_));
      NDLL_ENFORCE(type_idx_map_.insert({name, cpu_outputs_.size()-1}).second,
          "Output tensor meta insertion failed. Duplicate output name '" +
          name + "' exists.");

      cpu_output_info_.push_back(info);
      gpu_output_events_.push_back(EventList());
    } else {
      gpu_outputs_.push_back(TensorListPool<GPUBackend>(
              queue_depth_, batch_size_, bytes_per_sample_hint_));
      NDLL_ENFORCE(type_idx_map_.insert({name, gpu_outputs_.size()-1}).second,
          "Output tensor meta insertion failed. Duplicate output name '" +
          name + "' exists.");

      gpu_output_info_.push_back(info);
      gpu_output_events_.push_back(EventList(queue_depth_, &event_pool_));
    }
  }

  // All buffers start off as free
  for (int i = 0; i < queue_depth_; ++i) {
    free_queue_.push(i);
  }
}

void Executor::SetOutputBuffersForIter(int queue_idx, WorkspaceBlob *wsb) {
  // For each output, we need to hookup the next buffer
  // to the desired output workspaces, and also the
  // input workspaces of later ops that use them
  for (size_t i = 0; i < cpu_outputs_.size(); ++i) {
    auto &info = cpu_output_info_[i];
    NodeID node_id = info.prod_and_idx.first;
    int output_idx = info.prod_and_idx.second;
    NDLL_ENFORCE(graph_->NodeType(node_id) == NDLL_INTERNAL);
    
    int internal_op_id = graph_->NodeIdx(node_id);
    wsb->internal_op_data[internal_op_id].SetOutput(
        output_idx, cpu_outputs_[i].GetTL(queue_idx));

    for (size_t j = 0; j < info.con_and_idx.size(); ++j) {
      node_id = info.con_and_idx[j].first;
      int input_idx = info.con_and_idx[j].second;
      NDLL_ENFORCE(graph_->NodeType(node_id) == NDLL_GPU);

      int gpu_op_id = graph_->NodeIdx(node_id);
      wsb->gpu_op_data[gpu_op_id].SetInput(
          input_idx, cpu_outputs_[i].GetTL(queue_idx));
    }
  }

  for (size_t i = 0; i < gpu_outputs_.size(); ++i) {
    auto &info = gpu_output_info_[i];
    NodeID node_id = info.prod_and_idx.first;
    int output_idx = info.prod_and_idx.second;

    if (graph_->NodeType(node_id) == NDLL_INTERNAL) {
      int internal_op_id = graph_->NodeIdx(node_id);
      wsb->internal_op_data[internal_op_id].SetOutput(output_idx,
          gpu_outputs_[i].GetTL(queue_idx));
    } else if (graph_->NodeType(node_id) == NDLL_GPU) {
      int gpu_op_id = graph_->NodeIdx(node_id);
      wsb->gpu_op_data[gpu_op_id].SetOutput(output_idx,
          gpu_outputs_[i].GetTL(queue_idx));
    } else {
      NDLL_FAIL("Internal error. GPU output source is "
          "not gpu/internal op");
    }

    for (size_t j = 0; j < info.con_and_idx.size(); ++j) {
      node_id = info.con_and_idx[j].first;
      int input_idx = info.con_and_idx[j].second;
      NDLL_ENFORCE(graph_->NodeType(node_id) == NDLL_GPU);

      int gpu_op_id = graph_->NodeIdx(node_id);
      wsb->gpu_op_data[gpu_op_id].SetInput(input_idx,
          gpu_outputs_[i].GetTL(queue_idx));
    }
  }
}

} // namespace ndll
