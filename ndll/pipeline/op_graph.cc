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
    new_node->parents.push_back(parent_id);

    // Add new node as child
    auto &parent_node = this->node(parent_id);
    parent_node.children.push_back(new_node->id);

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
    auto ret = tensor_srcs_.insert({name, new_node->id});
    NDLL_ENFORCE(ret.second, "Operator '" + spec.name() +
        "' has output with name " + name + ", but output "
        "with this name already exists as output of op '" +
        this->node(TensorSourceID(name)).spec.name());
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

} // namespace ndll
