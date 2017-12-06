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
  PruneUnusedGraphNodes(graph, output_names);

  // Setup workspaces for each op and connect
  // their inputs and outputs.
  SetupDataForGraph(graph, &cpu_op_data_,
      &internal_op_data_, &gpu_op_data_);

  // Presize the workspaces based on the hint
  PresizeData(&cpu_op_data_, &internal_op_data_,
      &gpu_op_data_, bytes_per_sample_hint_);
  
  // Assign streams to all internal & gpu ops
  SetupStreamsForGraph(graph,
      &internal_op_data_, &gpu_op_data_,
      &stream_pool_, &event_pool_);

  // Setup queues and events for the outputs of each stage
  SetupOutputQueuesForGraph(output_names, queue_depth_,
      &event_pool_, graph, &type_idx_map_, &cpu_outputs_,
      &gpu_outputs_);
}

void Executor::RunCPU() {
  // Setup the output buffers for this iteration
  SetOutputBuffersForIter(output_names_, type_idx_map_,
      queue_idx_, graph_, &internal_op_data_, &gpu_op_data_,
      &cpu_outputs_, &gpu_outputs_);

  // TODO(tgale): Is this the desired behavior?
  // 
  // If we are about to issue work into a set of
  // buffers whose data has not been requested
  // by the user yet. Block until the user
  // queries for the data
  std::unique_lock<std::mutex> lock(ready_mutex_);
  while (!ready_queue_.empty() && ready_queue_.front() == queue_idx_) {
    ready_cond_.wait(lock);
  }
  lock.unlock();
  
  // Run the cpu-ops in the thread pool
  for (int i = 0; i < batch_size_; ++i) {
    thread_pool_.DoWorkWithID(std::bind(
            [this] (int data_idx, int tid) {
              SampleWorkspace ws;
              for (int j = 0; j < graph_->NumCPUOp(); ++j) {
                Operator<CPUBackend> &op = graph_->cpu_op(j);
                cpu_op_data_[j].GetSample(&ws, data_idx, tid);
                op.Run(&ws);
              }
            }, i, std::placeholders::_1));
  }
  thread_pool_.WaitForWork();
}

// TODO(tgale): Handle event insertion for internal
// and gpu operators. Setup events for output
// management and figure out mechanism through
// which user should query for outptus (especially
// old outputs.
void Executor::RunInternal() {
  for (int i = 0; i < graph_->NumInternalOp(); ++i) {
    internal::InternalOp &op = graph_->internal_op(i);
    internal::MixedWorkspace &ws = internal_op_data_[i];
    op.Run(&ws);
    if (ws.has_stream() && ws.has_event()) {
      CUDA_CALL(cudaEventRecord(ws.event(), ws.stream()));
    }
  }

  for (auto &sync_meta : internal_output_events_) {
    // Record the events that signal the completion
    // of the output data
    CUDA_CALL(cudaEventRecord(sync_meta.second, sync_meta.first));
  }
}

void Executor::RunGPU() {
  for (int i = 0; i < graph_->NumGPUOp(); ++i) {
    Operator<GPUBackend> &op = graph_->gpu_op(i);
    DeviceWorkspace &ws = gpu_op_data_[i];
    auto parent_events = ws.ParentEvents();
    for (auto &event : parent_events) {
      CUDA_CALL(cudaStreamWaitEvent(ws.stream(), event, 0));
    }
    op.Run(&ws);
    if (ws.has_event()) {
      CUDA_CALL(cudaEventRecord(ws.event(), ws.stream()));
    }
  }

  for (auto &sync_meta : gpu_output_events_) {
    // Record the events that signal the completion
    // of the output data
    CUDA_CALL(cudaEventRecord(sync_meta.second, sync_meta.first));
  }

  // Update the ready queue to signal that all the work
  // in the `queue_idx_` set of output buffers has been
  // issued. Notify any waiting threads.
  std::unique_lock<std::mutex> lock(ready_mutex_);
  ready_queue_.push(queue_idx_);
  ready_cond_.notify_one();
  lock.unlock();
  
  // Increment the queue index for next time.
  queue_idx_ = (queue_idx_ + 1) % queue_depth_;
}

void Executor::Outputs(DeviceWorkspace *ws) {
  NDLL_ENFORCE(ws != nullptr, "Workspace is nullptr");

  // Block until the work for a batch has been issued.
  // Once we get the index of our results, we notify
  // on the ready_cond in case the worker thread is
  // waiting to not overwrite these results
  std::unique_lock<std::mutex> lock(ready_mutex_);
  while (ready_queue_.empty()) {
    ready_cond_.wait(lock);
  }
  int output_idx = ready_queue_.front();
  ready_queue_.pop();
  ready_cond_.notify_one();
  lock.unlock();

  // Gather the results TensorLists and block on their
  // events to make sure that the computation has completed
  ws->Clear();
  for (const auto &name : output_names_) {
    auto it = type_idx_map_.find(name);
    NDLL_ENFORCE(it != type_idx_map_.end(), "Executor could not "
        "find output with name '" + name + "'.");

    if (graph_->TensorIsType<CPUBackend>(name)) {
      auto &tl_pool = cpu_outputs_[it->second];
      ws->AddOutput(tl_pool.GetTL(output_idx));
      CUDA_CALL(cudaEventSynchronize(tl_pool.GetEvent(output_idx)));
    } else {
      auto &tl_pool = gpu_outputs_[it->second];
      ws->AddOutput(tl_pool.GetTL(output_idx));
      CUDA_CALL(cudaEventSynchronize(tl_pool.GetEvent(output_idx)));
    }
  }
}

void Executor::PruneUnusedGraphNodes(OpGraph *graph,
    vector<string> output_names) {
  // We want to remove any nodes whose outputs are
  // never used by another node or as an output
  NDLL_ENFORCE(output_names.size() > 0,
      "No outputs requested, nothing to execute.");

  while (true) {
    // We do not edit the graph while we are iterating
    // as node ids will be updated when an op is removed
    vector<NodeID> to_remove;
    for (int i = 0; i < graph->NumOp(); ++i) {
      OpNode &node = graph->node(i);
      
      // If this node has children, don't prune it
      if (!node.children.empty()) continue;

      // Note: this is technically a very innefficient
      // way to find the insertsection of the node outputs
      // and the outputs of the graph. The number of outputs
      // is usually 1-2, so it shouldn't matter
      bool found_match = false;
      for (int j = 0; j < node.spec.NumOutput(); ++j) {
        for (size_t k = 0; k < output_names.size(); ++k) {
          if (node.spec.Output(j) == output_names[k]) {
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
      graph->RemoveOp(to_remove[i] - i);
    }
  }

  // If we've pruned the entire graph, something has gone wrong
  NDLL_ENFORCE(graph->NumOp() > 0, "No output names match "
      "data produced by the pipeline.");
}

void Executor::SetupDataForGraph(OpGraph *graph,
    vector<HostWorkspace> *cpu_data,
    vector<internal::MixedWorkspace> *internal_data,
    vector<DeviceWorkspace> *gpu_data) {
  // Clear any old data setup
  cpu_data->clear();
  internal_data->clear();
  gpu_data->clear();

  // Create workspaces for each operator
  cpu_data->resize(graph->NumCPUOp());
  internal_data->resize(graph->NumInternalOp());
  gpu_data->resize(graph->NumGPUOp());

  // Setup cpu op input and output buffers
  for (int i = 0; i < graph->NumCPUOp(); ++i) {
    CPUOpNode &node = graph->cpu_node(i);
    HostWorkspace &ws = (*cpu_data)[i];
    
    for (int j = 0; j < node.spec.NumInput(); ++j) {
      // Go get each set of input Tensors and add
      // them to this cpu ops workspace.
      NodeID parent_node_id = graph->TensorSourceID(node.spec.Input(j));
      NDLLOpType parent_op_type = graph->NodeType(parent_node_id);
      NDLL_ENFORCE(parent_op_type == NDLL_CPU,
          "Executor encountered cpu op with non-cpu input.");
      int parent_idx = graph->NodeIdx(parent_node_id);
      int input_src_idx = graph->TensorIdxInSource(node.spec.Input(j));

      HostWorkspace &src_ws = (*cpu_data)[parent_idx];
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
    internal::MixedWorkspace &ws = (*internal_data)[i];

    for (int j = 0; j < node.spec.NumInput(); ++j) {
      // Go get each set of input Tensors and add
      // them to this internal ops workspace.
      NodeID parent_node_id = graph->TensorSourceID(node.spec.Input(j));
      NDLLOpType parent_op_type = graph->NodeType(parent_node_id);
      NDLL_ENFORCE(parent_op_type == NDLL_CPU,
          "Executor encoutered internal op with non-cpu input.");
      int parent_idx = graph->NodeIdx(parent_node_id);
      int input_src_idx = graph->TensorIdxInSource(node.spec.Input(j));

      HostWorkspace &src_ws = (*cpu_data)[parent_idx];
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
    DeviceWorkspace &ws = (*gpu_data)[i];
    
    for (int j = 0; j < node.spec.NumInput(); ++j) {
      // Go get each set of input Tensors and add
      // them to this internal ops workspace.
      NodeID parent_node_id = graph->TensorSourceID(node.spec.Input(j));
      NDLLOpType parent_op_type = graph->NodeType(parent_node_id);
      int parent_idx = graph->NodeIdx(parent_node_id);
      int input_src_idx = graph->TensorIdxInSource(node.spec.Input(j));

      if (parent_op_type == NDLL_INTERNAL) {
        internal::MixedWorkspace &src_ws = (*internal_data)[parent_idx];
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
        DeviceWorkspace &src_ws = (*gpu_data)[parent_idx];
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

void Executor::PresizeData(
    vector<HostWorkspace> *cpu_data,
    vector<internal::MixedWorkspace> *internal_data,
    vector<DeviceWorkspace> *gpu_data,
    size_t bytes_per_sample_hint) {
  // Note: At some point our graph has source nodes that
  // only have outputs (data readers or external inputs).
  // Thus, the set of all outputs buffers in our workspaces
  // represents all the unique buffers in our graph.
  for (auto &ws : *cpu_data) {
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

  for (auto &ws : *internal_data) {
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
  
  for (auto &ws : *gpu_data) {
    for (int i = 0; i < ws.NumOutput(); ++i) {
      NDLL_ENFORCE(ws.OutputIsType<GPUBackend>(i), "Executor "
          "encountered gpu op with non-gpu output.");
      TensorList<GPUBackend> *tl = ws.Output<GPUBackend>(i);
      tl->mutable_data<uint8>();
      tl->Resize({{(Index)bytes_per_sample_hint_*batch_size_}});
    }
  }
}

void Executor::SetupStreamsForGraph(OpGraph *graph,
    vector<internal::MixedWorkspace> *internal_data,
    vector<DeviceWorkspace> *gpu_data,
    StreamPool *stream_pool, EventPool *event_pool) {

  for (int i = 0; i < graph->NumInternalOp(); ++i) {
    // For internal ops, we assign unique streams to each
    // op. This ensures (assuming the stream pool does not
    // have a limit) that we won't have false dependencies
    // between internal ops and the previous iterations
    // gpu ops.
    internal::MixedWorkspace &ws = (*internal_data)[i];
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

    // If this op has more than a single child node,
    // we will need an event to synchronize the child
    // that does not get the parents stream
    //
    // TODO(tgale): We could do this more efficiently by
    // applying the same algorithm we use to assign streams
    if (node.children.size() > 1) {
      DeviceWorkspace &ws = (*gpu_data)[graph->NodeIdx(i)];
      ws.set_event(event_pool_.GetEvent());
    }
    
    if (is_root) {
      node_queue.push(node.id);
      NDLL_ENFORCE(processed_nodes.insert(node.id).second);
    }
  }

  std::unordered_map<NodeID, cudaStream_t> node_streams;
  while (!node_queue.empty()) {
    const OpNode &current_node = graph->node(node_queue.front());
    int current_op_idx = graph->NodeIdx(current_node.id);
    DeviceWorkspace &ws = (*gpu_data)[current_op_idx];
    node_queue.pop();

    bool found_stream = false;
    auto it = current_node.parents.begin();
    for (; it != current_node.parents.end(); ++it) {
      NodeID parent_id = *it;
      int parent_op_idx = graph->NodeIdx(parent_id);

      if (graph->NodeType(parent_id) == NDLL_INTERNAL) {
        // We will not re-use internal op streams, but
        // we will need to block on this ops event to
        // make sure that we respect the dependency
        internal::MixedWorkspace parent_ws = (*internal_data)[parent_op_idx];
        ws.AddParentEvent(parent_ws.event());
      } else if (graph->NodeType(parent_id) == NDLL_GPU) {
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
        DeviceWorkspace parent_ws = (*gpu_data)[parent_op_idx];
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
      int parent_op_idx = graph->NodeIdx(parent_id);
      
      if (graph->NodeType(parent_id) == NDLL_INTERNAL) {
        internal::MixedWorkspace parent_ws = (*internal_data)[parent_op_idx];
        ws.AddParentEvent(parent_ws.event());
      } else if (graph->NodeType(parent_id) == NDLL_GPU) {
        DeviceWorkspace parent_ws = (*gpu_data)[parent_op_idx];
        ws.AddParentEvent(parent_ws.event());
      } else {
        NDLL_FAIL("Executor encountered gpu op with non-gpu/internal parent.");
      }
    }
    
    if (!found_stream) {
      // Couldn't reuse any parent streams, request
      // a new stream from the stream pool
      cudaStream_t stream = stream_pool->GetStream();
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

void Executor::SetupOutputQueuesForGraph(
    const vector<string> &output_names, int queue_depth,
    EventPool *event_pool, OpGraph *graph, 
    std::map<string, int> *type_idx_map,
    vector<TensorListPool<CPUBackend>> *cpu_outputs,
    vector<TensorListPool<GPUBackend>> *gpu_outputs) {
  // Allocate output TensorList pools for each output
  for (auto &name : output_names) {
    auto tensor_meta = graph->TensorSourceMeta(name);
    if (tensor_meta.is_cpu) {
      cpu_outputs->push_back(TensorListPool<CPUBackend>(queue_depth, event_pool));
      NDLL_ENFORCE(type_idx_map->insert({name, cpu_outputs->size()-1}).second,
          "Output tensor meta insertion failed. Duplicate output name '" +
          name + "' exists.");
    } else {
      gpu_outputs->push_back(TensorListPool<GPUBackend>(queue_depth, event_pool));
      NDLL_ENFORCE(type_idx_map->insert({name, gpu_outputs->size()-1}).second,
          "Output tensor meta insertion failed. Duplicate output name '" +
          name + "' exists.");
    }
  }
}

void Executor::SetOutputBuffersForIter(
    const vector<string> &output_names,
    const std::map<string, int> &type_idx_map,
    int queue_idx, OpGraph *graph,
    vector<internal::MixedWorkspace> *internal_data,
    vector<DeviceWorkspace> *gpu_data,
    vector<TensorListPool<CPUBackend>> *cpu_outputs,
    vector<TensorListPool<GPUBackend>> *gpu_outputs) {
  // For each output, we need to hookup the next buffer
  // to the desired output workspaces, and also the
  // input workspaces of later ops that use them
  internal_output_events_.clear();
  gpu_output_events_.clear();
  for (auto &name : output_names) {
    // Get the index of this output in whichever
    // tl type vector its stored in
    auto it = type_idx_map.find(name);
    NDLL_ENFORCE(it != type_idx_map.end(), "Could not "
        "find entry for tensor '" + name + "'.");
    int idx_in_typed_vec = it->second;

    auto tensor_meta = graph->TensorSourceMeta(name);
    NDLLOpType node_type = graph->NodeType(tensor_meta.node);
    if (node_type == NDLL_INTERNAL) {
      int op_idx = graph->NodeIdx(tensor_meta.node);
      internal::MixedWorkspace &ws = (*internal_data)[op_idx];
      if (tensor_meta.is_cpu) {
        auto &output = (*cpu_outputs)[idx_in_typed_vec];
        ws.SetOutput(tensor_meta.index, output.GetTL(queue_idx));
        
        if (ws.has_stream()) {
          // Mark this outputs event to be recorded
          cudaEvent_t event = output.GetEvent(queue_idx);
          internal_output_events_.push_back(
              std::make_pair(ws.stream(), event));
        }
      } else {
        auto &output = (*gpu_outputs)[idx_in_typed_vec];
        ws.SetOutput(tensor_meta.index, output.GetTL(queue_idx));

        if (ws.has_stream()) {
          // Mark this outputs event to be recorded
          cudaEvent_t event = output.GetEvent(queue_idx);
          internal_output_events_.push_back(
              std::make_pair(ws.stream(), event));
        }
      }
    } else if (node_type == NDLL_GPU) {
      NDLL_ENFORCE(!tensor_meta.is_cpu, "Executor "
          "encountered gpu op w/ cpu output");
      int op_idx = graph->NodeIdx(tensor_meta.node);
      DeviceWorkspace &ws = (*gpu_data)[op_idx];

      auto &output = (*gpu_outputs)[idx_in_typed_vec];
      ws.SetOutput(tensor_meta.index, output.GetTL(queue_idx));

      // Mark this outputs event to be recorded
      cudaEvent_t event = output.GetEvent(queue_idx);
      gpu_output_events_.push_back(
          std::make_pair(ws.stream(), event));
    } else {
      NDLL_FAIL("Output tensor with invalid type detected");
    }

    // Update all nodes workspaces that take this tensor as input
    vector<TensorMeta> consumer_meta = graph->TensorConsumerMeta(name);
    for (const auto &meta : consumer_meta) {
      // Ops that take in contiguous data are gpu ops
      NDLL_ENFORCE(graph->NodeType(meta.node) == NDLL_GPU,
          "Executor encountered contiguous input to non-gpu node.");

      DeviceWorkspace &ws = (*gpu_data)[graph->NodeIdx(meta.node)];
      if (meta.is_cpu) {
        auto output = (*cpu_outputs)[idx_in_typed_vec].GetTL(queue_idx);
        ws.SetOutput(meta.index, output);
      } else {
        auto output = (*gpu_outputs)[idx_in_typed_vec].GetTL(queue_idx);
        ws.SetOutput(meta.index, output);
      }
    }
  }
  
}

} // namespace ndll
