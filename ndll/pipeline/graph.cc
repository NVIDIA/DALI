#include "ndll/pipeline/graph.h"

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
    auto parent_id = GetTensorSource(spec.Input(i));
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
        nodes_[GetTensorSource(name)]->spec.name());
  }
  return node;
}

template <>
OpPtr<CPUBackend>& OpGraph::op(int idx) {
  NDLL_ENFORCE(idx >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)idx < nodes_.size(), "Index " + std::to_string(idx) +
      " out of range for graph with size " + std::to_string(nodes_.size()));
  NDLL_ENFORCE(idx < num_cpu_, "Op with given index does "
      "not have calling 'Backend' type.");
  return dynamic_cast<CPUOpNode*>(nodes_[idx].get())->op;
}

template <>
OpPtr<GPUBackend>& OpGraph::op(int idx) {
  NDLL_ENFORCE(idx >= 0, "Negative index not supported.");
  NDLL_ENFORCE((size_t)idx < nodes_.size(), "Index " + std::to_string(idx) +
      " out of range for graph with size " + std::to_string(nodes_.size()));
  NDLL_ENFORCE(idx >= num_cpu_, "Op with given index does "
      "not have calling 'Backend' type.");
  return dynamic_cast<GPUOpNode*>(nodes_[idx].get())->op;
}

} // namespace ndll
