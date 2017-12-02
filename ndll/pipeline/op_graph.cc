#include "ndll/pipeline/op_graph.h"

namespace ndll {

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

template <typename Backend>
void CheckOpConstraints(const OpSpec &spec, const OpPtr<Backend> &op) {
  NDLL_ENFORCE(op->SupportsInPlace() || !spec.GetArgument<bool>("inplace", false),
      "Op '" + spec.name() + "' does not support in-place execution.");
  NDLL_ENFORCE(spec.NumInput() <= op->MaxNumInput(), "Operator '" + spec.name() +
      "' supports a maximum of " + std::to_string(op->MaxNumInput()) + " inputs, "
      "but was passed " + std::to_string(spec.NumInput()) + ".");
  NDLL_ENFORCE(spec.NumInput() >= op->MinNumInput(), "Operator '" + spec.name() +
      "' supports a minimum of " + std::to_string(op->MinNumInput()) + " inputs, "
      "but was passed " + std::to_string(spec.NumInput()) + ".");
  NDLL_ENFORCE(spec.NumOutput() <= op->MaxNumOutput(), "Operator '" + spec.name() +
      "' supports a maximum of " + std::to_string(op->MaxNumOutput()) + " outputs, "
      "but was passed " + std::to_string(spec.NumOutput()) + ".");
  NDLL_ENFORCE(spec.NumOutput() >= op->MinNumOutput(), "Operator '" + spec.name() +
      "' supports a minimum of " + std::to_string(op->MinNumOutput()) + " outputs, "
        "but was passed " + std::to_string(spec.NumOutput()) + ".");
}

} // namespace

void OpGraph::AddOp(const OpSpec &spec) {
  string device = spec.GetArgument<string>("device", "cpu");
  OpNode *new_node;
  if (device == "cpu") {
    // Enforce graph constraints
    NDLL_ENFORCE(AllInputsCPU(spec), "CPU ops cannot receive GPU input data.");
    NDLL_ENFORCE(AllOutputsCPU(spec), "CPU ops can only produce CPU output data.");

    // Create the operator
    OpPtr<CPUBackend> tmp(
        CPUOperatorRegistry::Registry().Create(spec.name(), spec));

    // Validate the number of inputs and execution settings
    CheckOpConstraints(spec, tmp);
      
    cpu_nodes_.resize(cpu_nodes_.size()+1);
    CPUOpNode &cpu_node = cpu_nodes_.back();
    cpu_node.op = std::move(tmp);
    id_to_node_map_.push_back({NDLL_CPU, cpu_nodes_.size()-1});

    new_node = &cpu_node;
  } else if (device == "gpu") {
    // Enforce graph constraints
    NDLL_ENFORCE(AllOutputsGPU(spec), "GPU ops can only produce GPU output data.");
    
    // Create the operator
    OpPtr<GPUBackend> tmp(
        GPUOperatorRegistry::Registry().Create(spec.name(), spec));

    // Validate the number of inputs and execution settings
    CheckOpConstraints(spec, tmp);

    gpu_nodes_.resize(gpu_nodes_.size()+1);
    GPUOpNode &gpu_node = gpu_nodes_.back();
    gpu_node.op = std::move(tmp);
    id_to_node_map_.push_back({NDLL_GPU, gpu_nodes_.size()-1});

    new_node = &gpu_node;
  } else if (device == "internal") {
    // Enforce graph constraints
    NDLL_ENFORCE(AllInputsCPU(spec), "Internal ops cannot receive GPU input data.");
    
    // Create the operator
    unique_ptr<internal::InternalOp> tmp(
        internal::InternalOpRegistry::Registry().Create(spec.name(), spec));

    internal_nodes_.resize(internal_nodes_.size()+1);
    InternalOpNode &internal_node = internal_nodes_.back();
    internal_node.op = std::move(tmp);
    id_to_node_map_.push_back({NDLL_INTERNAL, internal_nodes_.size()-1});

    new_node = &internal_node;
  } else {
    NDLL_FAIL("Invalid device argument \"" + device +
        "\". Valid options are \"cpu\", \"gpu\" or \"internal\"");
  }

  // Add node meta-data and add to the list of nodes
  new_node->id = NumOp()-1;
  new_node->spec = spec;

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

    // Save the id of the parent node and the index of
    // this nodes input in the set of the parents outputs
    string input_name = spec.InputName(i);
    string input_device = spec.InputDevice(i);
    int input_idx = parent_node.spec.OutputIdxForName(input_name, input_device);
    new_node->input_src_and_idx.push_back(std::make_pair(parent_node.id, input_idx));
  }

  // Mark this op as the source of its output tensors
  for (int i = 0; i < spec.NumOutput(); ++i) {
    string name = spec.Output(i);

    TensorMeta meta;
    meta.source = new_node->id;
    meta.idx_in_source = i;
    meta.is_cpu = spec.OutputDevice(i) == "cpu" ? true : false;
    
    auto ret = tensor_srcs_.insert({name, meta});
    NDLL_ENFORCE(ret.second, "Operator '" + spec.name() +
        "' has output with name " + name + ", but output "
        "with this name already exists as output of op '" +
        this->node(TensorSourceID(name)).spec.name());
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
  NDLL_ENFORCE(target.children.empty(), "Node '" + target.spec.name() +
      "' has " + std::to_string(target.children.size()) +
      ". Cannot remove");

  // Remove this nodes tensors from the graph
  for (int i = 0; i < target.spec.NumOutput(); ++i) {
    tensor_srcs_.erase(target.spec.Output(i));
  }

  for (int i = 0; i < this->NumOp(); ++i) {
    OpNode &node = this->node(i);
    if (i > id) {
      // Decrement this nodes id to account for
      // the removal of the node with id `id`.
      --node.id;

      // Update all of its outputs with the new id
      for (int j = 0; j < node.spec.NumOutput(); ++j) {
        auto it = tensor_srcs_.find(node.spec.Output(j));
        NDLL_ENFORCE(it != tensor_srcs_.end(),
            "Could not find tensor source entry.");
        
        it->second.source = node.id;
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
      NDLL_ENFORCE(*it != id, "Found node with target as parent.");
      if (*it > id) {
        to_add.push_back((*it) - 1);
        it = node.parents.erase(it);
      } else {
        ++it;
      }
    }
    for (auto &parent : to_add) {
      NDLL_ENFORCE(node.parents.insert(parent).second,
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
      NDLL_ENFORCE(node.children.insert(child).second,
          "Insertion of updated child id failed.");
    }

  }
  
  // Remove this nodes entry from the id map. This will
  // effectively decrement all node ids after this node
  // to fill the gap.
  //
  // TODO(tgale): Node remove is relatively expensive
  // because we use a vector. Consider switching to
  // a map if this becomes an issue.
  auto type_and_idx = id_to_node_map_[id];
  NDLLOpType type = type_and_idx.first;
  int idx = type_and_idx.second;
  id_to_node_map_.erase(id_to_node_map_.begin() + id);
  
  // Remove the typed node object for the target node.
  // We will then need to update the id map entry for
  // all nodes of this type that follow the deleted node
  switch (type) {
  case NDLL_CPU:
    cpu_nodes_.erase(cpu_nodes_.begin() + idx);

    for (size_t i = idx; i < cpu_nodes_.size(); ++i) {
      CPUOpNode &cpu_node = this->cpu_node(i);
      id_to_node_map_[cpu_node.id].second = i;
    }
    break;
  case NDLL_GPU:
    gpu_nodes_.erase(gpu_nodes_.begin() + idx);

    for (size_t i = idx; i < gpu_nodes_.size(); ++i) {
      GPUOpNode &gpu_node = this->gpu_node(i);
      id_to_node_map_[gpu_node.id].second = i;
    }
    break;
  case NDLL_INTERNAL:
    internal_nodes_.erase(internal_nodes_.begin() + idx);

    for (size_t i = idx; i < internal_nodes_.size(); ++i) {
      InternalOpNode &internal_node = this->internal_node(i);
      id_to_node_map_[internal_node.id].second = i;
    }
    break;
  }
}

OpNode& OpGraph::node(NodeID id) {
  NDLL_ENFORCE_VALID_INDEX((size_t)id, id_to_node_map_.size());
  auto idx_pair = id_to_node_map_[id];

  switch (idx_pair.first) {
  case NDLL_CPU:
    return cpu_nodes_[idx_pair.second];
    break;
  case NDLL_GPU:
    return gpu_nodes_[idx_pair.second];
    break;
  case NDLL_INTERNAL:
    return internal_nodes_[idx_pair.second];
    break;    
  default:
    NDLL_FAIL("Internal error. Invalid node type index.");
  }
}

template <>
bool OpGraph::TensorIsType<CPUBackend>(const string &name) {
  return TensorInfo(name).is_cpu;
}

template <>
bool OpGraph::TensorIsType<GPUBackend>(const string &name) {
  return !TensorInfo(name).is_cpu;
}

} // namespace ndll
