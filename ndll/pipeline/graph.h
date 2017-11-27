#ifndef NDLL_PIPELINE_OP_GRAPH_H_
#define NDLL_PIPELINE_OP_GRAPH_H_

#include <map>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/operator.h"

namespace ndll {

template <typename T>
using OpPtr = unique_ptr<Operator<T>>;

typedef int64 NodeID;

struct OpNode {
  inline OpNode() {}
  virtual ~OpNode() = default;

  NodeID id;
  OpSpec spec;
  vector<NodeID> parents, children;
};

struct CPUOpNode : public OpNode {
  OpPtr<CPUBackend> op;
};

struct GPUOpNode : public OpNode {
  OpPtr<GPUBackend> op;
};

class OpGraph {
public:
  inline OpGraph() : num_cpu_(0) {}
  ~OpGraph() = default;

  void AddOp(const OpSpec &spec) {
    auto node = CreateNode(spec);
    nodes_.push_back(node);
  }

  inline NodeID GetTensorSource(const string &name) {
    auto it = tensor_srcs_.find(name);
    NDLL_ENFORCE(it != tensor_srcs_.end(), "Tensor with name \"" +
        name + "\" has no know source.");
    return it->second;
  }

  inline int NumCPUOp() const { return num_cpu_; }

  inline int NumGPUOp() const { return nodes_.size() - num_cpu_; }

  template <typename Backend>
  OpPtr<Backend>& op(int idx);
  
  DISABLE_COPY_MOVE_ASSIGN(OpGraph);
private:
  shared_ptr<OpNode> CreateNode(const OpSpec &spec);

  vector<shared_ptr<OpNode>> nodes_;
  int num_cpu_;

  std::map<string, NodeID> tensor_srcs_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_OP_GRAPH_H_
