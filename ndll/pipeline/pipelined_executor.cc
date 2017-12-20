#include "ndll/pipeline/pipelined_executor.h"

#include <set>
#include <utility>

namespace ndll {

void PipelinedExecutor::Build(OpGraph *graph, vector<string> output_names) {
  Executor::Build(graph, output_names);

  // Setup queues for the intermediate outputs of each stage
  SetupStageOutputsForGraph();
}

void PipelinedExecutor::SetupStageOutputsForGraph() {
  // Make a set of the outputs names for quick lookup
  std::set<string> output_set(output_names_.begin(), output_names_.end());
  
  for (int i = 0; i < graph_->NumCPUOp(); ++i) {
    // Find all outputs of the cpu stage. An output is
    // a tensor that is used by an op in a later stage.
    CPUOpNode &node = graph_->cpu_node(i);
    for (int j = 0; j < node.spec.NumOutput(); ++j) {
      // If this tensor is a pipeline output, its
      // queueing will be handled by the Executor base
      string tensor_name = node.spec.Output(j);
      if (output_set.count(tensor_name) != 0) continue;
      
      vector<TensorMeta> consumer_meta =
        graph_->TensorConsumerMeta(tensor_name);
      bool has_info_object = false;
      for (auto &meta : consumer_meta) {
        if (graph_->NodeType(meta.node) != NDLL_CPU) {
          // We've located a tensor that is an output of
          // the stage. Save the index of this cpu node,
          // the index of this tensor in its workspace,
          // the index of all non-cpu node consumers,
          // and the index of this tensor in the consumer
          // workspaces.
          if (!has_info_object) {
            OutputInfo info;
            info.prod_and_idx = std::make_pair(node.id, j);
            cpu_stage_output_info_.push_back(info);
            cpu_stage_outputs_.push_back(
                TensorVectorPool<CPUBackend>(
                    queue_depth_, batch_size_, bytes_per_sample_hint_
                    ));
            has_info_object = true;
          }

          OutputInfo &info = cpu_stage_output_info_.back();
          auto tmp = std::make_pair(meta.node, meta.index);
          info.con_and_idx.push_back(tmp);
        }
      }
    }
  }

  for (int i = 0; i < graph_->NumInternalOp(); ++i) {
    // Find all outputs of the internal stage. An output
    // is a tensor that is used by an op in a later stage.
    InternalOpNode &node = graph_->internal_node(i);
    for (int j = 0; j < node.spec.NumOutput(); ++j) {
      // If this tensor is a pipeline output, its
      // queueing will be handled by the Executor base
      string tensor_name = node.spec.Output(j);
      if (output_set.count(tensor_name) != 0) continue;

      vector<TensorMeta> consumer_meta =
        graph_->TensorConsumerMeta(tensor_name);
      bool has_info_object = false;
      for (auto &meta : consumer_meta) {
        if (graph_->NodeType(meta.node) != NDLL_INTERNAL) {
          if (!has_info_object) {
            OutputInfo info;
            info.prod_and_idx = std::make_pair(node.id, j);
            internal_stage_output_info_.push_back(info);
            internal_stage_outputs_.push_back(
                TensorListPool<GPUBackend>(
                    queue_depth_, batch_size_, bytes_per_sample_hint_
                    ));
            has_info_object = true;
          }

          
          OutputInfo &info = internal_stage_output_info_.back();
          auto tmp = std::make_pair(meta.node, meta.index);
          info.con_and_idx.push_back(tmp);
        }
      }
    }
  }
} 

void PipelinedExecutor::SetStageOutputsForIter(
    int queue_idx, WorkspaceBlob *wsb) {
  for (size_t i = 0; i < cpu_stage_outputs_.size(); ++i) {
    auto &tvp = cpu_stage_outputs_[i];
    auto &info = cpu_stage_output_info_[i];
    NodeID node_id = info.prod_and_idx.first;
    NDLL_ENFORCE(graph_->NodeType(node_id) == NDLL_CPU);
    
    int cpu_op_id = graph_->NodeIdx(node_id);
    int output_idx = info.prod_and_idx.second;
    wsb->cpu_op_data[cpu_op_id].SetOutput(
        output_idx, tvp.GetTV(queue_idx));

    for (size_t j = 0; j < info.con_and_idx.size(); ++j) {
      node_id = info.con_and_idx[j].first;
      NDLL_ENFORCE(graph_->NodeType(node_id) == NDLL_INTERNAL);

      int internal_op_id = graph_->NodeIdx(node_id);
      int input_idx = info.con_and_idx[j].second;
      wsb->internal_op_data[internal_op_id].SetInput(
          input_idx, tvp.GetTV(queue_idx));
    }
  }

  for (size_t i = 0; i < internal_stage_outputs_.size(); ++i) {
    auto &tlp = internal_stage_outputs_[i];
    auto &info = internal_stage_output_info_[i];
    NodeID node_id = info.prod_and_idx.first;
    NDLL_ENFORCE(graph_->NodeType(node_id) == NDLL_INTERNAL);
    
    int internal_op_id = graph_->NodeIdx(node_id);
    int output_idx = info.prod_and_idx.second;
    wsb->internal_op_data[internal_op_id].SetOutput(
        output_idx, tlp.GetTL(queue_idx));
    
    for (size_t j = 0; j < info.con_and_idx.size(); ++j) {
      node_id = info.con_and_idx[j].first;
      NDLL_ENFORCE(graph_->NodeType(node_id) == NDLL_GPU);
      
       int gpu_op_id = graph_->NodeIdx(node_id);
      int input_idx = info.con_and_idx[j].second;
      wsb->gpu_op_data[gpu_op_id].SetInput(
          input_idx, tlp.GetTL(queue_idx));
    }
  }
}

} // namespace ndll

