// Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/pipeline/graph/op_graph.h"

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

// TODO(klecki): Graph creation is not a place to check OpSpec?
void CheckOpConstraints(const OpSpec &spec) {
  const OpSchema &schema = SchemaRegistry::GetSchema(spec.name());

  bool allows_multiple_inputs_sets = schema.AllowsMultipleInputSets();
  const int additional_outputs = schema.CalculateAdditionalOutputs(spec);

  int num_input_sets = 1;
  if (allows_multiple_inputs_sets) {
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
      "Operator '" + spec.name() + "' supports "
      + std::to_string((schema.CalculateOutputs(spec) + additional_outputs)/num_input_sets)
      + " outputs, but was passed " + std::to_string(spec.NumOutput()/num_input_sets) + ".");
}

OpType ParseOpType(const std::string &device) {
  if (device == "gpu") {
    return OpType::GPU;
  } else if (device == "cpu") {
    return OpType::CPU;
  } else if (device == "mixed") {
    return OpType::MIXED;
  } else if (device == "support") {
    return OpType::SUPPORT;
  }
  DALI_FAIL("Unsupported device type: " + device + ".");
}

StorageDevice ParseStorageDevice(const std::string &io_device) {
  if (io_device == "cpu") {
    return StorageDevice::CPU;
  }
  return StorageDevice::GPU;
}

}  // namespace

OpNode& OpGraph::PlaceNewOp(OpType op_type, OpSpec op_spec, std::string instance_name) {
  op_nodes_.push_back(OpNode());
  auto &node = op_nodes_.back();
  node.id = op_nodes_.size() - 1;
  node.spec = op_spec;
  node.instance_name = std::move(instance_name);
  node.op_type = op_type;
  auto new_partition_id = NumOp(op_type);
  node.partition_index = new_partition_id;
  op_partitions_[static_cast<int>(op_type)].push_back(node.id);
  return node;
}

TensorNode& OpGraph::PlaceNewTensor() {
  tensor_nodes_.push_back(TensorNode());
  tensor_nodes_.back().id = tensor_nodes_.size() - 1;
  return tensor_nodes_.back();
}

void OpGraph::AddOp(const OpSpec &spec, const std::string& name) {
  // Validate the op specification
  CheckOpConstraints(spec);

  string device = spec.GetArgument<string>("device");
  auto op_type = ParseOpType(device);
  // TODO(klecki): refactor this out
  switch (op_type) {
    case OpType::CPU: {
      // Enforce graph constraints
      DALI_ENFORCE(AllInputsCPU(spec), "CPU ops cannot receive GPU input data.");
      DALI_ENFORCE(AllOutputsCPU(spec), "CPU ops can only produce CPU output data.");
      break;
    }
    case OpType::GPU: {
      break;
    }
    case OpType::MIXED: {
      // Enforce graph constraints
      DALI_ENFORCE(AllInputsCPU(spec), "Mixed ops cannot receive GPU input data.");
      break;
    }
    case OpType::SUPPORT: {
      // Enforce graph constraints
      DALI_ENFORCE(AllInputsCPU(spec), "Support ops cannot receive GPU input data.");
      break;
    }
    default:
      DALI_FAIL("Invalid device argument \"" + device +
          "\". Valid options are \"cpu\", \"gpu\" or \"mixed\"");
      break;
  }
  // Add node meta-data and add to the list of nodes
  auto &new_node = PlaceNewOp(op_type, spec, name);

  // Setup references between nodes. We require that the
  // ops are added to the graph in a topological ordering.
  // This loop will verify this by ensuring that all inputs
  // to the new op have already been created.
  for (int i = 0; i < spec.NumInput(); ++i) {
    // Get the tensor id we are consuming by its name
    auto it = tensor_name_to_id_.find(spec.Input(i));
    DALI_ENFORCE(it != tensor_name_to_id_.end(), "Tensor with name \"" +
        name + "\" has no known source.");
    auto consumed_tensor_id = it->second;

    // Add parent node id, checks if parent node exists in graph
    auto parent_id = tensor_nodes_[consumed_tensor_id].producer.node;

    // Note: We don't care if the parent has already
    // been added to this nodes set of parents, so
    // we don't check the return value.
    new_node.parents.insert(parent_id);

    // Add new node as child
    auto &parent_node = this->Node(parent_id);
    parent_node.children.insert(new_node.id);

    // Place it as a parent tensor
    new_node.parent_tensors.push_back(consumed_tensor_id);

    // Update the consumer info for this tensor
    TensorMeta meta;
    meta.node = new_node.id;
    meta.index = i;
    meta.is_support = spec.IsArgumentInput(i);
    meta.storage_device = ParseStorageDevice(spec.InputDevice(i));

    // Insert new tensor consumer
    tensor_nodes_[consumed_tensor_id].consumers.push_back(meta);
  }

  // Mark this op as the source of its output tensors
  // Create TensorNodes for outputs and respective edges for this OpNode -> TensorNodes
  for (int i = 0; i < spec.NumOutput(); ++i) {
    // Set the producer info for this tensor
    TensorMeta meta;
    meta.node = new_node.id;
    meta.index = i;
    meta.is_support = spec.GetArgument<string>("device") == "support";
    meta.storage_device = ParseStorageDevice(spec.OutputDevice(i));

    string name = spec.Output(i);

    // Place new Tensor with producer info and add edge to OpNode.
    auto &new_tensor = PlaceNewTensor();
    new_tensor.producer = meta;
    new_tensor.name = name;
    new_node.children_tensors.push_back(new_tensor.id);

    auto it_inserted = tensor_name_to_id_.insert({name, new_tensor.id});
    DALI_ENFORCE(it_inserted.second, "Operator '" + spec.name() +
        "' has output with name " + name + ", but output "
        "with this name already exists as output of op '" +
        this->Node(TensorSourceID(name)).spec.name() + "'");
  }
}

void OpGraph::InstantiateOperators() {
  // traverse devices by topological order (support, cpu, mixed, gpu)
  OpType order[] = {OpType::SUPPORT, OpType::CPU, OpType::MIXED, OpType::GPU};

  for (auto op_type : order) {
    for (auto op_id : op_partitions_[static_cast<int>(op_type)]) {
      op_nodes_[op_id].InstantiateOperator();
    }
  }
}

void OpGraph::SwapTensorNodes(TensorNodeId left_id, TensorNodeId right_id) {
  auto &left = tensor_nodes_[left_id];
  auto &right = tensor_nodes_[right_id];
  // Change ids in producers (there is only one) of left
  auto &left_prod = Node(left.producer.node);
  left_prod.children_tensors[left.producer.index] = right_id;
  // Change ids in producers (there is only one) of right
  auto &right_prod = Node(right.producer.node);
  right_prod.children_tensors[right.producer.index] = left_id;
  // Change ids in consumers of left node to right id
  for (auto &cons_edge : left.consumers) {
    auto &cons = Node(cons_edge.node);
    cons.parent_tensors[cons_edge.index] = right_id;
  }
  // Change ids in consumers of right node to left id
  for (auto &cons_edge : right.consumers) {
    auto &cons = Node(cons_edge.node);
    cons.parent_tensors[cons_edge.index] = left_id;
  }
  // Clean up names mapping
  tensor_name_to_id_[left.name] = right_id;
  tensor_name_to_id_[right.name] = left_id;
  // Swap the actual nodes
  left.id = right_id;
  right.id = left_id;
  std::swap(left, right);
}

void OpGraph::RemoveTensorNode(TensorNodeId id) {
  DALI_ENFORCE_VALID_INDEX(id, (Index)tensor_nodes_.size());
  DALI_ENFORCE(tensor_nodes_[id].consumers.empty(),
               "Removed tensors cannot have any consumers.");
  auto removed_name = tensor_nodes_[id].name;
  // Swap it out
  for (TensorNodeId i = id + 1; i < static_cast<int>(tensor_nodes_.size()); i++) {
    // Move from i to i - 1
    SwapTensorNodes(i, i - 1);
  }
  // We remove the last element
  tensor_nodes_.pop_back();
  tensor_name_to_id_.erase(removed_name);
  // There is no option to remove from positional array of tensor produced by parent op
}

void OpGraph::SwapOpNodes(OpNodeId left_id, OpNodeId right_id) {
  auto &left = op_nodes_[left_id];
  auto &right = op_nodes_[right_id];
  // Swap all references in tensor edges
  // Produced tensors (children)
  {
    auto &tensor_nodes_ref = tensor_nodes_;
    auto swap_ids_in_parent_tensor = [&tensor_nodes_ref](OpNode &node, OpNodeId new_id) {
      for (auto tensor_id : node.children_tensors) {
        auto &tensor = tensor_nodes_ref[tensor_id];
        // edges from node to tensor
        tensor.producer.node = new_id;
      }
    };
    swap_ids_in_parent_tensor(left, right_id);
    swap_ids_in_parent_tensor(right, left_id);
  }
  // Consumed tensors (parents). As we can have overlapping parents, we do this in two steps
  // otherwise we could overwrite twice.
  {
    auto &tensor_nodes_ref = tensor_nodes_;
    auto swap_ids_in_child_tensor = [&tensor_nodes_ref](OpNode &node, OpNodeId old_id,
                                                        OpNodeId new_id) {
      for (auto tensor_id : node.parent_tensors) {
        auto &tensor = tensor_nodes_ref[tensor_id];
        // edges from tensor to node
        for (auto &edge : tensor.consumers) {
          if (edge.node == old_id) {
            edge.node = new_id;
          }
        }
      }
    };
    constexpr OpNodeId dummy_id = -1;
    swap_ids_in_child_tensor(left, left_id, dummy_id);
    swap_ids_in_child_tensor(right, right_id, left_id);
    swap_ids_in_child_tensor(left, dummy_id, right_id);
  }
  // Swap all references in parent and children ops
  {
    auto &op_nodes_ref = op_nodes_;
    auto remove_id_in_family_op = [&op_nodes_ref](OpNode &node, OpNodeId old_id) {
      for (auto oid : node.parents) {
        op_nodes_ref[oid].children.erase(old_id);
      }
      for (auto oid : node.children) {
        op_nodes_ref[oid].parents.erase(old_id);
      }
    };
    auto add_id_in_family_op = [&op_nodes_ref](OpNode &node, OpNodeId new_id) {
      for (auto oid : node.parents) {
        op_nodes_ref[oid].children.insert(new_id);
      }
      for (auto oid : node.children) {
        op_nodes_ref[oid].parents.insert(new_id);
      }
    };
    remove_id_in_family_op(left, left_id);
    remove_id_in_family_op(right, right_id);
    add_id_in_family_op(left, right_id);
    add_id_in_family_op(right, left_id);
  }

  // Swap the nodes
  left.id = right_id;
  right.id = left_id;
  std::swap(left, right);
}

void OpGraph::RemoveOpNode(OpNodeId id) {
  DALI_ENFORCE_VALID_INDEX(id, (Index)op_nodes_.size());
  auto &target_op = op_nodes_[id];
  DALI_ENFORCE(target_op.children.empty(), "Overwritten ops cannot have any children.");
  DALI_ENFORCE(target_op.children_tensors.empty(),
               "All produced tensors should be removed before removing op"
               " and list of children tensors should be invalidated.");
  for (OpNodeId i = id + 1; i < static_cast<int>(op_nodes_.size()); i++) {
    // Move from i to i - 1
    SwapOpNodes(i - 1, i);
  }
  // Remove the edge from parent Ops
  for (auto parent_id : op_nodes_.back().parents) {
    Node(parent_id).children.erase(op_nodes_.back().id);
  }
  // assume that we removed one element
  op_nodes_.pop_back();
}

namespace {

/**
 * @brief Removes element from `index` by swapping with back and poping
 *
 * Does not maintain the order of elements in vector
 * @tparam T
 * @param vector
 * @param index
 */
template <typename T>
void RemoveVectorElement(T& vector, int index) {
  std::swap(vector[index], vector.back());
  vector.pop_back();
}

}  // namespace

// Op Removal Process:
// 1. Validate we can remove it (it has no children & no consumers for produced tensors)
// 2. Remove tensors it produces
// 3. Remove us from consumer lists of parent tensors
// 4. Remove the OpNode with edges from parent Ops
// 6. Correct the partitions as we changed the ids while removing OpNode
void OpGraph::RemoveOp(OpNodeId id) {
  OpNode &target = this->Node(id);

  // If the node has any children, we cannot remove it
  DALI_ENFORCE(target.children.empty(), "Node '" + target.spec.name() +
      "' has " + std::to_string(target.children.size()) +
      ". Cannot remove");
  for (auto t : target.children_tensors) {
    DALI_ENFORCE(tensor_nodes_[t].consumers.empty(), "Node '" + target.spec.name() +
      "' produces a tensor that has " + std::to_string(tensor_nodes_[t].consumers.size()) +
      " consumers. Cannot remove");
  }

  // Remove all tensors produced by this node and invalidate list of children tensors
  for (auto t : target.children_tensors) {
    RemoveTensorNode(t);
  }
  target.children_tensors.clear();

  // In case we consume this tensor more than once, we try to remove all occurences
  for (auto t : target.parent_tensors) {
    auto &sibling_consumers = tensor_nodes_[t].consumers;
    for (size_t i = 0; i < sibling_consumers.size(); i++) {
      if (sibling_consumers[i].node == id) {
        RemoveVectorElement(sibling_consumers, i);
      }
    }
  }

  RemoveOpNode(id);
  // Just recalculate, do not try to fix
  RepartitionOps();
}

void OpGraph::RepartitionOps() {
  for (auto & p : op_partitions_) {
    p.clear();
  }
  for (auto &node : op_nodes_) {
    auto new_partition_id = NumOp(node.op_type);
    node.partition_index = new_partition_id;
    op_partitions_[static_cast<int>(node.op_type)].push_back(node.id);
  }
}

std::vector<std::vector<TensorNodeId>> OpGraph::PartitionTensorByOpType() const {
  std::vector<std::vector<TensorNodeId>> out;
  out.resize(static_cast<int>(OpType::COUNT));
  for (auto &tensor : tensor_nodes_) {
    auto producer_op_type = Node(tensor.producer.node).op_type;
    out[static_cast<int>(producer_op_type)].push_back(tensor.id);
  }
  return out;
}

// TODO(klecki): get rid of string indexing
OpNode& OpGraph::Node(const std::string& name) {
  for (auto &node : op_nodes_) {
    if (node.instance_name == name) {
      return node;
    }
  }
  DALI_FAIL("Operator node with name " + name + " not found.");
}

namespace {
  /**
   * @brief Prints instance_name of OpNode to stream
   *
   * @param ofs
   * @param node
   * @param show_ids Whether to print name concatenated with `_id`.
   * @return std::ofstream&
   */
  std::ofstream& PrintTo(std::ofstream &ofs, const OpNode& node, bool show_ids) {
    ofs << node.instance_name;
    if (show_ids) {
      ofs << "_" << node.id;
    }
    return ofs;
  }
  /**
   * @brief Prints TensorNode's name to stream
   *
   * @param ofs
   * @param node
   * @param show_ids Whether to print name concatenated with `_id`.
   * @return std::ofstream&
   */
  std::ofstream&  PrintTo(std::ofstream &ofs, const TensorNode& node, bool show_ids) {
    ofs << node.name;
    if (show_ids) {
      ofs << "_" << node.id;
    }
    return ofs;
  }
  std::string GetOpColor(OpType op_type) {
    switch (op_type) {
      case OpType::CPU:
        return "blue";
      case OpType::GPU:
        return "#76b900";
      case OpType::MIXED:
        return "cyan";
      case OpType::SUPPORT:
        return "grey";
      default:
        return "black";
    }
  }
}  // namespace

void OpGraph::GenerateDOTFromGraph(std::ofstream &ofs, bool show_tensors, bool show_ids,
                                   bool use_colors) {
  // Just output all the edges
  for (auto &op : op_nodes_) {
    if (use_colors) {
      PrintTo(ofs, op, show_ids) << "[color=\"" << GetOpColor(op.op_type) << "\"];\n";
    }
    for (auto child_id : op.children) {
      auto& child_node = Node(child_id);
      PrintTo(ofs, op, show_ids) << " -> ";
      PrintTo(ofs, child_node, show_ids);
      if (show_tensors) {
        ofs << "[style=dotted]";
      }
      ofs << ";\n";
    }
    if (show_tensors) {
      int i = 0;
      for (auto t_id : op.children_tensors) {
        TensorNode& child_tensor = Tensor(t_id);
        PrintTo(ofs, child_tensor, show_ids) << "[shape=box];\n";
        PrintTo(ofs, op, show_ids) << " -> ";
        PrintTo(ofs, child_tensor, show_ids) << "[label=" << i++ <<"];\n";
        GenerateDOTFromGraph(child_tensor, ofs, show_tensors, show_ids);
      }
    }
    ofs << "\n";
  }
}

void OpGraph::GenerateDOTFromGraph(const TensorNode &current_node, std::ofstream &ofs,
                                   bool show_tensors, bool show_ids) {
  for (auto edge : current_node.consumers) {
      PrintTo(ofs, current_node, show_ids) << " -> ";
      auto &child_op = Node(edge.node);
      PrintTo(ofs, child_op, show_ids) << "[label=" << edge.index << "];\n";
  }
}

std::vector<TensorNodeId> OpGraph::GetOutputs(const std::vector<string>& output_names) const {
  std::vector<TensorNodeId> result;
  for (const auto& out : output_names) {
    result.push_back(TensorId(out));
  }
  return result;
}

std::vector<TensorNodeId> OpGraph::GetStageOutputs(OpType stage) const {
  std::vector<TensorNodeId> result;
  for (const auto& tensor : tensor_nodes_) {
    // Check if the tensor is produced in current stage
    if (Node(tensor.producer.node).op_type == stage) {
      for (auto cons_edge : tensor.consumers) {
        // We found a consumer from different stage, this tensor is a stage output
        if (Node(cons_edge.node).op_type != stage) {
          result.push_back(tensor.id);
          break;
        }
      }
    }
  }
  return result;
}


template <>
bool OpGraph::TensorIsType<CPUBackend>(const string &name) {
  return TensorSourceMeta(name).storage_device == StorageDevice::CPU;
}

template <>
bool OpGraph::TensorIsType<GPUBackend>(const string &name) {
  return TensorSourceMeta(name).storage_device == StorageDevice::GPU;
}

}  // namespace dali
