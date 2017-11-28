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

  /**
   * @brief Adds an op with the input specification to the graph.
   */
  inline void AddOp(const OpSpec &spec) {
    auto node = CreateNode(spec);
    nodes_.push_back(node);
  }

  /**
   * @brief Returns the id of the op that produces the tensor with
   * the given name.
   */
  inline NodeID TensorSourceID(const string &name) {
    auto it = tensor_srcs_.find(name);
    NDLL_ENFORCE(it != tensor_srcs_.end(), "Tensor with name \"" +
        name + "\" has no know source.");
    return it->second;
  }

  /**
   * @brief Returns the total number of ops in the graph.
   */
  inline int NumOp() const { return nodes_.size(); }
  
  /**
   * @brief Returns the number of ops in the graph with the given 
   * backend.
   */
  template <typename Backend>
  int NumOpWithBackend() const;

  /**
   * @brief Returns the operator with the given index in the graph.
   */
  template <typename Backend>
  OpPtr<Backend>& op(NodeID id);

  /**
   * @brief Returns the graph node with the given index in the graph.
   */
  OpNode& node(NodeID id);
  
  DISABLE_COPY_MOVE_ASSIGN(OpGraph);
private:
  shared_ptr<OpNode> CreateNode(const OpSpec &spec);

  vector<shared_ptr<OpNode>> nodes_;
  int num_cpu_;

  std::map<string, NodeID> tensor_srcs_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_OP_GRAPH_H_
