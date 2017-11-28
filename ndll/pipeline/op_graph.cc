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

shared_ptr<OpNode> OpGraph::CreateNode(const OpSpec &spec) {
  string device = spec.GetArgument<string>("device", "cpu");
  shared_ptr<OpNode> node;
  if (device == "cpu") {
    // Enforce graph constraints
    NDLL_ENFORCE((size_t)num_cpu_ == nodes_.size(), "All CPU operators "
        "must occur before any GPU operators.");
    NDLL_ENFORCE(AllInputsCPU(spec));
    NDLL_ENFORCE(AllOutputsCPU(spec));

    // Create the operator
    OpPtr<CPUBackend> tmp(
        CPUOperatorRegistry::Registry().Create(spec.name(), spec));

    // Validate the number of inputs and execution settings
    CheckOpConstraints(spec, tmp);
      
    CPUOpNode *ptr = new CPUOpNode;
    ptr->op = std::move(tmp);
    node.reset(ptr);
    ++num_cpu_;
  } else if (device == "gpu") {
    // Enforce graph constraints
    NDLL_ENFORCE(AllOutputsGPU(spec));
    
    // Create the operator
    OpPtr<GPUBackend> tmp(
        GPUOperatorRegistry::Registry().Create(spec.name(), spec));

    // Validate the number of inputs and execution settings
    CheckOpConstraints(spec, tmp);
    
    GPUOpNode *ptr = new GPUOpNode;
    ptr->op = std::move(tmp);
    node.reset(ptr);
  } else {
    NDLL_FAIL("Invalid device argument \"" + device +
        "\". Valid options are \"cpu\" or \"gpu\"");
  }

  // Add node meta-data and add to the list of nodes
  node->id = nodes_.size();
  node->spec = spec;

  // Setup references between nodes. We require that the
  // ops are added to the graph in a topological ordering.
  // This loop will verify this by ensuring that all inputs
  // to the new op have already been created.
  for (int i = 0; i < spec.NumInput(); ++i) {
    // Add parent node id
    auto parent_id = TensorSourceID(spec.Input(i));
    node->parents.push_back(parent_id);

    // Add new node as child
    auto &parent_node = nodes_[parent_id];
    parent_node->children.push_back(node->id);
  }

  // Mark this op as the source of its output tensors
  for (int i = 0; i < spec.NumOutput(); ++i) {
    string name = spec.Output(i);
    auto ret = tensor_srcs_.insert({name, node->id});
    NDLL_ENFORCE(ret.second, "Operator '" + spec.name() +
        "' has output with name " + name + ", but output "
        "with this name already exists as output of op '" +
        nodes_[TensorSourceID(name)]->spec.name());
  }
  return node;
}

template <>
int OpGraph::NumOpWithBackend<CPUBackend>() const {
  return num_cpu_;
}

template <>
int OpGraph::NumOpWithBackend<GPUBackend>() const {
  return nodes_.size() - num_cpu_;
}

template <>
OpPtr<CPUBackend>& OpGraph::op(NodeID id) {
  NDLL_ENFORCE(id >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)id < nodes_.size(), "Index " + std::to_string(id) +
      " out of range for graph with size " + std::to_string(nodes_.size()));
  NDLL_ENFORCE(id < num_cpu_, "Op with given index does "
      "not have calling 'Backend' type.");
  return dynamic_cast<CPUOpNode*>(nodes_[id].get())->op;
}

template <>
OpPtr<GPUBackend>& OpGraph::op(NodeID id) {
  NDLL_ENFORCE(id >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)id < nodes_.size(), "Index " + std::to_string(id) +
      " out of range for graph with size " + std::to_string(nodes_.size()));
  NDLL_ENFORCE(id >= num_cpu_, "Op with given index does "
      "not have calling 'Backend' type.");
  return dynamic_cast<GPUOpNode*>(nodes_[id].get())->op;
}

OpNode& OpGraph::node(NodeID id) {
  NDLL_ENFORCE(id >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)id < nodes_.size(), "Index " + std::to_string(id) +
      " out of range for graph with size " + std::to_string(nodes_.size()));
  return *nodes_[id];
}

} // namespace ndll
