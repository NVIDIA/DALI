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

#include "dali/pipeline/op_graph.h"

#include "dali/pipeline/operators/op_schema.h"

namespace dali {

namespace {

bool AllInputsCPU(const OpSpec &spec) {
  for (int i = 0; i < spec.NumInput(); ++i) {
    if (spec.InputDevice(i) == "gpu") return false;
  }
  return true;
}

bool AllOutputsCPU(const OpSpec &spec) {
  for (int i = 0; i < spec.NumOutput(); ++i) {
    if (spec.OutputDevice(i) == "gpu") return false;
  }
  return true;
}

bool AllOutputsGPU(const OpSpec &spec) {
  for (int i = 0; i < spec.NumOutput(); ++i) {
    if (spec.OutputDevice(i) == "cpu") return false;
  }
  return true;
}

void CheckOpConstraints(const OpSpec &spec) {
  OpSchema schema = SchemaRegistry::GetSchema(spec.name());

  bool allows_multiple_inputs = schema.AllowsMultipleInputSets();
  const int additional_outputs = schema.CalculateAdditionalOutputs(spec);

  int num_input_sets = 1;
  if (allows_multiple_inputs) {
    num_input_sets = spec.GetArgument<int>("num_input_sets");
  } else {
    DALI_ENFORCE(spec.GetArgument<int>("num_input_sets") == 1,
        "Op '" + spec.name() + "' does not support multiple input sets.");
  }

  DALI_ENFORCE(schema.SupportsInPlace(spec) || !spec.GetArgument<bool>("inplace"),
      "Op '" + spec.name() + "' does not support in-place execution.");
  DALI_ENFORCE(spec.NumRegularInput() <= num_input_sets * schema.MaxNumInput(),
      "Operator '" + spec.name() +
      "' supports a maximum of " + std::to_string(schema.MaxNumInput()) + " inputs, "
      "but was passed " + std::to_string(spec.NumRegularInput()) + ".");
  DALI_ENFORCE(spec.NumRegularInput() >= num_input_sets * schema.MinNumInput(),
      "Operator '" + spec.name() +
      "' supports a minimum of " + std::to_string(schema.MinNumInput()) + " inputs, "
      "but was passed " + std::to_string(spec.NumRegularInput()) + ".");
  DALI_ENFORCE(spec.NumOutput() == schema.CalculateOutputs(spec) + additional_outputs,
      "Operator '" + spec.name() +
      "' supports " + std::to_string(schema.CalculateOutputs(spec)/num_input_sets) + " outputs, "
      "but was passed " + std::to_string(spec.NumOutput()/num_input_sets) + ".");
}

}  // namespace

void OpGraph::AddOp(const OpSpec &spec, const std::string& name) {
  // Validate the op specification
  CheckOpConstraints(spec);

  string device = spec.GetArgument<string>("device");
  OpNode *new_node;
  if (device == "cpu") {
    // Enforce graph constraints
    DALI_ENFORCE(AllInputsCPU(spec), "CPU ops cannot receive GPU input data.");
    DALI_ENFORCE(AllOutputsCPU(spec), "CPU ops can only produce CPU output data.");

    // Create the operator
    OpPtr tmp(
        CPUOperatorRegistry::Registry().Create(spec.name(), spec, &device));

    cpu_nodes_.resize(cpu_nodes_.size()+1);
    OpNode &cpu_node = cpu_nodes_.back();
    cpu_node.op = std::move(tmp);
    id_to_node_map_.push_back({DALI_CPU, cpu_nodes_.size()-1});

    new_node = &cpu_node;
  } else if (device == "gpu") {
    // Create the operator
    OpPtr tmp(
        GPUOperatorRegistry::Registry().Create(spec.name(), spec, &device));

    gpu_nodes_.resize(gpu_nodes_.size()+1);
    OpNode &gpu_node = gpu_nodes_.back();
    gpu_node.op = std::move(tmp);
    id_to_node_map_.push_back({DALI_GPU, gpu_nodes_.size()-1});

    new_node = &gpu_node;
  } else if (device == "mixed") {
    // Enforce graph constraints
    DALI_ENFORCE(AllInputsCPU(spec), "Mixed ops cannot receive GPU input data.");

    // Create the operator
    OpPtr tmp(
        MixedOperatorRegistry::Registry().Create(spec.name(), spec, &device));

    mixed_nodes_.resize(mixed_nodes_.size()+1);
    OpNode &mixed_node = mixed_nodes_.back();
    mixed_node.op = std::move(tmp);
    id_to_node_map_.push_back({DALI_MIXED, mixed_nodes_.size()-1});

    new_node = &mixed_node;
  } else if (device == "support") {
    // Enforce graph constraints
    DALI_ENFORCE(AllInputsCPU(spec), "Support ops cannot receive GPU input data.");

    // Create the operator
    OpPtr tmp(
        SupportOperatorRegistry::Registry().Create(spec.name(), spec, &device));

    support_nodes_.resize(support_nodes_.size()+1);
    OpNode &support_node = support_nodes_.back();
    support_node.op = std::move(tmp);
    id_to_node_map_.push_back({DALI_SUPPORT, support_nodes_.size() - 1});

    new_node = &support_node;
  } else {
    DALI_FAIL("Invalid device argument \"" + device +
        "\". Valid options are \"cpu\", \"gpu\" or \"mixed\"");
  }

  // Add node meta-data and add to the list of nodes
  new_node->id = NumOp()-1;
  new_node->spec = spec;
  new_node->instance_name = name;

  // Setup references between nodes. We require that the
  // ops are added to the graph in a topological ordering.
  // This loop will verify this by ensuring that all inputs
  // to the new op have already been created.
  for (int i = 0; i < spec.NumInput(); ++i) {
    // Add parent node id
    auto parent_id = TensorSourceID(spec.Input(i));

    // Note: We don't care if the parent has already
    // been added to this nodes set of parents, so
    // we don't check the return value.
    new_node->parents.insert(parent_id);

    // Add new node as child
    auto &parent_node = this->node(parent_id);
    parent_node.children.insert(new_node->id);

    // Update the consumer info for this tensor
    TensorMeta meta;
    meta.node = new_node->id;
    meta.index = i;
    meta.is_support = spec.IsArgumentInput(i);
    meta.is_cpu = spec.InputDevice(i) == "cpu" ? true : false;

    vector<TensorMeta> &consumer_info = tensor_consumers_[spec.Input(i)];
    consumer_info.push_back(meta);
  }

  // Mark this op as the source of its output tensors
  for (int i = 0; i < spec.NumOutput(); ++i) {
    string name = spec.Output(i);

    // Set the producer info for this tensor
    TensorMeta meta;
    meta.node = new_node->id;
    meta.index = i;
    meta.is_support = spec.GetArgument<string>("device") == "support";
    meta.is_cpu = spec.OutputDevice(i) == "cpu" ? true : false;

    auto ret = tensor_producers_.insert({name, meta});
    DALI_ENFORCE(ret.second, "Operator '" + spec.name() +
        "' has output with name " + name + ", but output "
        "with this name already exists as output of op '" +
        this->node(TensorSourceID(name)).spec.name() + "'");
  }
}

// Op Removal Process:
// 1. Validate we can remove it (it has no children)
// 2. Remove its tensors
// 3. Remove it as a child of all ops
// 4. Decrement all child ids > id
// 5. Decrement all parent ids > id
// 5. Decrement all op ids > id
// 6. remove id map entry for target
// 7. remove object for target
// 8. update id map for ops after target in its typed vector
void OpGraph::RemoveOp(NodeID id) {
  OpNode &target = this->node(id);

  // If the node has any children, we cannot remove it
  DALI_ENFORCE(target.children.empty(), "Node '" + target.spec.name() +
      "' has " + std::to_string(target.children.size()) +
      ". Cannot remove");

  // Remove this nodes tensors from the graph
  for (int i = 0; i < target.spec.NumOutput(); ++i) {
    tensor_producers_.erase(target.spec.Output(i));
  }

  // Remove references to this node as a consumer
  for (int i = 0; i < target.spec.NumInput(); ++i) {
    auto it = tensor_consumers_.find(target.spec.Input(i));
    DALI_ENFORCE(it != tensor_consumers_.end(), "Could not find "
        "consumer entries for tensor, but target node is a consumer.");
    vector<TensorMeta> &consumer_info = it->second;
    bool erased = false;
    for (size_t j = 0; j < consumer_info.size(); ++j) {
      if (consumer_info[j].node == id) {
        consumer_info.erase(consumer_info.begin() + j);
        erased = true;
        break;
      }
    }
    DALI_ENFORCE(erased, "Could not find entry for target node as tensor consumer.");
  }

  for (int i = 0; i < this->NumOp(); ++i) {
    OpNode &node = this->node(i);
    if (node.id > id) {
      // Decrement this nodes id to account for
      // the removal of the node with id `id`.
      --node.id;

      // Update all of its outputs with the new id
      for (int j = 0; j < node.spec.NumOutput(); ++j) {
        auto it = tensor_producers_.find(node.spec.Output(j));
        DALI_ENFORCE(it != tensor_producers_.end(),
            "Could not find tensor source entry.");

        it->second.node = node.id;
      }

      // Update all of its consumer records with new id
      for (int j = 0; j < node.spec.NumInput(); ++j) {
        auto it = tensor_consumers_.find(node.spec.Input(j));
        DALI_ENFORCE(it != tensor_consumers_.end(), "Could not find "
            "consumer entries for tensor, but current node is a consumer.");
        vector<TensorMeta> &consumer_info = it->second;
        bool found = false;
        for (size_t k = 0; k < consumer_info.size(); ++k) {
          if (consumer_info[k].node == node.id+1) {
            consumer_info[k].node = node.id;
            found = true;
            break;
          }
        }
        DALI_ENFORCE(found, "Could not find entry for current "
            "node as tensor consumer.");
      }
    }

    // Scan its parents and children. If the target is
    // a child, remove it as it no longer exists. If
    // a node with an id > the target id is a parent
    // or child, we will decrement its id to account
    // for the removal.
    vector<NodeID> to_add;
    auto it = node.parents.begin();
    while (it != node.parents.end()) {
      // This should never occur, we have previously checked
      // that the target has no children in the graph
      DALI_ENFORCE(*it != id, "Found node with target as parent.");
      if (*it > id) {
        to_add.push_back((*it) - 1);
        it = node.parents.erase(it);
      } else {
        ++it;
      }
    }
    for (auto &parent : to_add) {
      DALI_ENFORCE(node.parents.insert(parent).second,
          "Insertion of updated parent id failed.");
    }
    to_add.clear();

    // Remove the target node id if it is a child
    node.children.erase(id);
    it = node.children.begin();
    while (it != node.children.end()) {
      if (*it > id) {
        to_add.push_back((*it) - 1);
        it = node.children.erase(it);
      } else {
        ++it;
      }
    }
    for (auto &child : to_add) {
      DALI_ENFORCE(node.children.insert(child).second,
          "Insertion of updated child id failed.");
    }
  }

  // Remove this nodes entry from the id map. This will
  // effectively decrement all node ids after this node
  // to fill the gap.
  //
  auto type_and_idx = id_to_node_map_[id];
  DALIOpType type = type_and_idx.first;
  int idx = type_and_idx.second;
  id_to_node_map_.erase(id_to_node_map_.begin() + id);

  // Remove the typed node object for the target node.
  // We will then need to update the id map entry for
  // all nodes of this type that follow the deleted node
  switch (type) {
  case DALI_CPU:
    cpu_nodes_.erase(cpu_nodes_.begin() + idx);

    for (size_t i = idx; i < cpu_nodes_.size(); ++i) {
      OpNode &cpu_node = this->cpu_node(i);
      id_to_node_map_[cpu_node.id].second = i;
    }
    break;
  case DALI_GPU:
    gpu_nodes_.erase(gpu_nodes_.begin() + idx);

    for (size_t i = idx; i < gpu_nodes_.size(); ++i) {
      OpNode &gpu_node = this->gpu_node(i);
      id_to_node_map_[gpu_node.id].second = i;
    }
    break;
  case DALI_MIXED:
    mixed_nodes_.erase(mixed_nodes_.begin() + idx);

    for (size_t i = idx; i < mixed_nodes_.size(); ++i) {
      OpNode &mixed_node = this->mixed_node(i);
      id_to_node_map_[mixed_node.id].second = i;
    }
    break;
  case DALI_SUPPORT:
    support_nodes_.erase(support_nodes_.begin() + idx);

    for (size_t i = idx; i < support_nodes_.size(); ++i) {
      OpNode &support_node = this->support_node(i);
      id_to_node_map_[support_node.id].second = i;
    }
  }
}

OpNode& OpGraph::node(NodeID id) {
  DALI_ENFORCE_VALID_INDEX(id, id_to_node_map_.size());
  auto idx_pair = id_to_node_map_[id];

  switch (idx_pair.first) {
  case DALI_CPU:
    return cpu_nodes_[idx_pair.second];
    break;
  case DALI_GPU:
    return gpu_nodes_[idx_pair.second];
    break;
  case DALI_MIXED:
    return mixed_nodes_[idx_pair.second];
    break;
  case DALI_SUPPORT:
    return support_nodes_[idx_pair.second];
    break;
  default:
    DALI_FAIL("Internal error. Invalid node type index.");
  }
}

OpNode& OpGraph::node(const std::string& name) {
  // Search cpu nodes
  for (auto& node : cpu_nodes_) {
    if (node.instance_name == name) {
      return node;
    }
  }
  // Search gpu nodes
  for (auto& node : gpu_nodes_) {
    if (node.instance_name == name) {
      return node;
    }
  }
  // Search mixed nodes
  for (auto& node : mixed_nodes_) {
    if (node.instance_name == name) {
      return node;
    }
  }
  // Search support nodes
  for (auto& node : support_nodes_) {
    if (node.instance_name == name) {
      return node;
    }
  }
  DALI_FAIL("Operator node with name " + name + " not found.");
}

template <>
bool OpGraph::TensorIsType<CPUBackend>(const string &name) {
  return TensorSourceMeta(name).is_cpu;
}

template <>
bool OpGraph::TensorIsType<GPUBackend>(const string &name) {
  return !TensorSourceMeta(name).is_cpu;
}

}  // namespace dali
