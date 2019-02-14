// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <algorithm>
#include <set>
#include <string>
#include <utility>
#include <vector>
#include "dali/pipeline/executor/pipelined_executor.h"
#include "dali/pipeline/operators/common.h"

namespace dali {

void PipelinedExecutor::Build(OpGraph *graph, vector<string> output_names) {
  Executor::Build(graph, output_names);

  // Setup queues for the intermediate outputs of each stage
  SetupStageOutputsForGraph();

  // For each set of outputs, link the correct stage
  // output buffer to each workspace.
  for (int i = 0; i < queue_depth_; ++i) {
    SetStageOutputsForIter(i, &wss_[i]);
  }
}

std::vector<int> PipelinedExecutor::GetMemoryHints(const OpSpec &spec) {
  std::vector<int> hints;
  GetSingleOrRepeatedArg(spec, &hints, "bytes_per_sample_hint", spec.NumOutput());
  std::replace(hints.begin(), hints.end(), 0, static_cast<int>(bytes_per_sample_hint_));
  return hints;
}

void PipelinedExecutor::SetupStageOutputsForGraph() {
  DeviceGuard g(device_id_);
  // Make a set of the outputs names for quick lookup
  std::set<string> output_set(output_names_.begin(), output_names_.end());

  for (int i = 0; i < graph_->NumOp(DALIOpType::DALI_SUPPORT); ++i) {
    // Find all outputs of the support stage. An output is
    // a tensor that is used by an op in a later stage.
    // Do not include CPU ops inputs, since those are run
    // synchronously with support ops
    OpNode &node = graph_->Node(DALIOpType::DALI_SUPPORT, i);
    std::vector<int> hints = GetMemoryHints(node.spec);

    for (int j = 0; j < node.spec.NumOutput(); ++j) {
      // If this tensor is a pipeline output, its
      // queueing will be handled by the Executor base
      string tensor_name = node.spec.Output(j);
      DALI_ENFORCE(graph_->TensorIsType<CPUBackend>(tensor_name));
      if (output_set.count(tensor_name) != 0) continue;

      vector<TensorMeta> consumer_meta =
        graph_->TensorConsumerMeta(tensor_name);
      bool found_stage_boundary = false;
      for (auto &meta : consumer_meta) {
        const auto& node_type = graph_->NodeType(meta.node);
        if (node_type != DALIOpType::DALI_SUPPORT &&
            node_type != DALIOpType::DALI_CPU) {
          // We've located a tensor that is an output of
          // the stage.
          found_stage_boundary = true;
        }
      }
      if (found_stage_boundary) {
        OutputInfo info;
        info.prod_and_idx = std::make_pair(node.id, j);
        support_stage_output_info_.push_back(info);
        support_stage_outputs_.push_back(
            TensorPool<CPUBackend>(
                queue_depth_, batch_size_, hints[j]));
        for (auto &meta : consumer_meta) {
          OutputInfo &info = support_stage_output_info_.back();
          auto tmp = std::make_pair(meta.node, meta.index);
          info.con_and_idx.push_back(tmp);
        }
      }
    }
  }

  for (int i = 0; i < graph_->NumOp(DALIOpType::DALI_CPU); ++i) {
    // Find all outputs of the cpu stage. An output is
    // a tensor that is used by an op in a later stage.
    OpNode &node = graph_->Node(DALIOpType::DALI_CPU, i);
    std::vector<int> hints = GetMemoryHints(node.spec);

    for (int j = 0; j < node.spec.NumOutput(); ++j) {
      // If this tensor is a pipeline output, its
      // queueing will be handled by the Executor base
      string tensor_name = node.spec.Output(j);
      DALI_ENFORCE(graph_->TensorIsType<CPUBackend>(tensor_name));
      if (output_set.count(tensor_name) != 0) continue;

      vector<TensorMeta> consumer_meta =
        graph_->TensorConsumerMeta(tensor_name);
      bool found_stage_boundary = false;
      bool has_gpu_consumer = false;
      for (auto &meta : consumer_meta) {
        auto type = graph_->NodeType(meta.node);
        if (type != DALIOpType::DALI_CPU) {
          // We've located a tensor that is an output of
          // the stage.
          auto &consumer = graph_->Node(meta.node);
          has_gpu_consumer =
              has_gpu_consumer ||
              type == DALIOpType::DALI_GPU ||
              (consumer.spec.name() == "MakeContiguous" &&
              consumer.spec.OutputDevice(0) == "gpu");
          found_stage_boundary = true;
        }
      }
      if (found_stage_boundary) {
        OutputInfo info;
        info.prod_and_idx = std::make_pair(node.id, j);
        cpu_stage_output_info_.push_back(info);
        cpu_stage_outputs_.push_back(
            TensorVectorPool<CPUBackend>(
                queue_depth_, batch_size_, hints[j], has_gpu_consumer));
        for (auto &meta : consumer_meta) {
          OutputInfo &info = cpu_stage_output_info_.back();
          auto tmp = std::make_pair(meta.node, meta.index);
          info.con_and_idx.push_back(tmp);
        }
      }
    }
  }

  for (int i = 0; i < graph_->NumOp(DALIOpType::DALI_MIXED); ++i) {
    // Find all outputs of the mixed stage. An output
    // is a tensor that is used by an op in a later stage.
    OpNode &node = graph_->Node(DALIOpType::DALI_MIXED, i);
    std::vector<int> hints = GetMemoryHints(node.spec);

    for (int j = 0; j < node.spec.NumOutput(); ++j) {
      // If this tensor is a pipeline output, its
      // queueing will be handled by the Executor base
      string tensor_name = node.spec.Output(j);
      if (output_set.count(tensor_name) != 0) continue;

      vector<TensorMeta> consumer_meta =
        graph_->TensorConsumerMeta(tensor_name);
      bool has_info_object = false;

      if (graph_->TensorIsType<CPUBackend>(tensor_name)) {
        for (auto &meta : consumer_meta) {
          if (graph_->NodeType(meta.node) != DALIOpType::DALI_MIXED) {
            if (!has_info_object) {
              OutputInfo info;
              info.prod_and_idx = std::make_pair(node.id, j);

              bool pinned = graph_->NodeType(meta.node) == DALIOpType::DALI_GPU;

              mixed_stage_cpu_output_info_.push_back(info);
              mixed_stage_cpu_outputs_.push_back(
                  TensorListPool<CPUBackend>(
                      queue_depth_, batch_size_, hints[j], pinned));
              has_info_object = true;
            }

            OutputInfo &info = mixed_stage_cpu_output_info_.back();
            auto tmp = std::make_pair(meta.node, meta.index);
            info.con_and_idx.push_back(tmp);
          }
        }
      } else {
        for (auto &meta : consumer_meta) {
          if (graph_->NodeType(meta.node) != DALIOpType::DALI_MIXED) {
            if (!has_info_object) {
              OutputInfo info;
              info.prod_and_idx = std::make_pair(node.id, j);
              mixed_stage_gpu_output_info_.push_back(info);
              mixed_stage_gpu_outputs_.push_back(
                  TensorListPool<GPUBackend>(
                      queue_depth_, batch_size_, hints[j]));
              has_info_object = true;
            }

            OutputInfo &info = mixed_stage_gpu_output_info_.back();
            auto tmp = std::make_pair(meta.node, meta.index);
            info.con_and_idx.push_back(tmp);
          }
        }
      }
    }
  }
}

void PipelinedExecutor::SetStageOutputsForIter(
    int queue_idx, WorkspaceBlob *wsb) {
  DeviceGuard g(device_id_);
  for (size_t i = 0; i < support_stage_outputs_.size(); ++i) {
    auto &tvp = support_stage_outputs_[i];
    auto &info = support_stage_output_info_[i];
    OpNodeId node_id = info.prod_and_idx.first;
    DALI_ENFORCE(graph_->NodeType(node_id) == DALIOpType::DALI_SUPPORT);

    int support_op_id = graph_->NodeIdx(node_id);
    int output_idx = info.prod_and_idx.second;
    wsb->support_op_data[support_op_id].SetOutput(
        output_idx, tvp.Get(queue_idx));

    for (size_t j = 0; j < info.con_and_idx.size(); ++j) {
      node_id = info.con_and_idx[j].first;
      const OpNode& op_node = graph_->Node(node_id);
      int child_op_id = graph_->NodeIdx(node_id);
      int input_idx = info.con_and_idx[j].second;
      const OpSpec& spec = op_node.spec;
      std::string arg_name = spec.ArgumentInputName(input_idx);
      if (graph_->NodeType(node_id) == DALIOpType::DALI_MIXED) {
        wsb->mixed_op_data[child_op_id].SetArgumentInput(
          tvp.Get(queue_idx), arg_name);
      } else if (graph_->NodeType(node_id) == DALIOpType::DALI_GPU) {
        wsb->gpu_op_data[child_op_id].SetArgumentInput(
          tvp.Get(queue_idx), arg_name);
      } else {
        wsb->cpu_op_data[child_op_id].SetArgumentInput(
          tvp.Get(queue_idx), arg_name);
      }
    }
  }

  for (size_t i = 0; i < cpu_stage_outputs_.size(); ++i) {
    auto &tvp = cpu_stage_outputs_[i];
    auto &info = cpu_stage_output_info_[i];
    OpNodeId node_id = info.prod_and_idx.first;
    DALI_ENFORCE(graph_->NodeType(node_id) == DALIOpType::DALI_CPU);

    int cpu_op_id = graph_->NodeIdx(node_id);
    int output_idx = info.prod_and_idx.second;
    wsb->cpu_op_data[cpu_op_id].SetOutput(
        output_idx, tvp.Get(queue_idx));

    for (size_t j = 0; j < info.con_and_idx.size(); ++j) {
      node_id = info.con_and_idx[j].first;
      if (graph_->NodeType(node_id) == DALIOpType::DALI_MIXED) {
        int mixed_op_id = graph_->NodeIdx(node_id);
        int input_idx = info.con_and_idx[j].second;
        wsb->mixed_op_data[mixed_op_id].SetInput(
          input_idx, tvp.Get(queue_idx));
        const OpNode &node = graph_->Node(DALIOpType::DALI_MIXED, mixed_op_id);
        std::vector<int> hints = GetMemoryHints(node.spec);
        // Use pinned memory only when it is useful
        if (node.spec.name() == "MakeContiguous" &&
            node.spec.NumOutput() == 1 &&
            node.spec.OutputDevice(0) == "gpu") {
          for (auto& v : tvp.Get(queue_idx)) {
            if (!v->is_pinned()) {
              auto capacity = std::max<size_t>(v->capacity(), hints[0]);
              v->free();
              v->set_pinned(true);
              v->reserve(capacity);
            }
          }
        }
      } else if (graph_->NodeType(node_id) == DALIOpType::DALI_CPU) {
        int cpu_op_id = graph_->NodeIdx(node_id);
        int input_idx = info.con_and_idx[j].second;
        wsb->cpu_op_data[cpu_op_id].SetInput(
          input_idx, tvp.Get(queue_idx));
      } else {
          DALI_FAIL("Internal error - found non-CPU/mixed consumer");
      }
    }
  }

  for (size_t i = 0; i < mixed_stage_cpu_outputs_.size(); ++i) {
    auto &tlp = mixed_stage_cpu_outputs_[i];
    auto &info = mixed_stage_cpu_output_info_[i];
    OpNodeId node_id = info.prod_and_idx.first;
    DALI_ENFORCE(graph_->NodeType(node_id) == DALIOpType::DALI_MIXED);

    int mixed_op_id = graph_->NodeIdx(node_id);
    int output_idx = info.prod_and_idx.second;
    wsb->mixed_op_data[mixed_op_id].SetOutput(
        output_idx, tlp.Get(queue_idx));

    for (size_t j = 0; j < info.con_and_idx.size(); ++j) {
      node_id = info.con_and_idx[j].first;
      DALI_ENFORCE(graph_->NodeType(node_id) == DALIOpType::DALI_GPU);

      int gpu_op_id = graph_->NodeIdx(node_id);
      int input_idx = info.con_and_idx[j].second;
      wsb->gpu_op_data[gpu_op_id].SetInput(
          input_idx, tlp.Get(queue_idx));
    }
  }


  for (size_t i = 0; i < mixed_stage_gpu_outputs_.size(); ++i) {
    auto &tlp = mixed_stage_gpu_outputs_[i];
    auto &info = mixed_stage_gpu_output_info_[i];
    OpNodeId node_id = info.prod_and_idx.first;
    DALI_ENFORCE(graph_->NodeType(node_id) == DALIOpType::DALI_MIXED);

    int mixed_op_id = graph_->NodeIdx(node_id);
    int output_idx = info.prod_and_idx.second;
    wsb->mixed_op_data[mixed_op_id].SetOutput(
        output_idx, tlp.Get(queue_idx));

    for (size_t j = 0; j < info.con_and_idx.size(); ++j) {
      node_id = info.con_and_idx[j].first;
      DALI_ENFORCE(graph_->NodeType(node_id) == DALIOpType::DALI_GPU);

      int gpu_op_id = graph_->NodeIdx(node_id);
      int input_idx = info.con_and_idx[j].second;
      wsb->gpu_op_data[gpu_op_id].SetInput(
          input_idx, tlp.Get(queue_idx));
    }
  }
}

}  // namespace dali

