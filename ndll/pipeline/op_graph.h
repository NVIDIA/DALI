#ifndef NDLL_PIPELINE_OP_GRAPH_H_
#define NDLL_PIPELINE_OP_GRAPH_H_

#include <map>

#include "ndll/common.h"
#include "ndll/error_handling.h"
#include "ndll/pipeline/internal_op.h"
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

struct InternalOpNode : public OpNode {
  unique_ptr<internal::InternalOp> op;
};

class OpGraph {
public:
  inline OpGraph() {}
  inline ~OpGraph() = default;

  /**
   * @brief Adds an op with the input specification to the graph.
   */
  void AddOp(const OpSpec &spec);

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
  inline int NumOp() const {
    return NumCPUOp() + NumGPUOp() + NumInternalOp();
  }
  
  /**
   * @brief Returns the number of cpu ops in the graph.
   */
  inline int NumCPUOp() const { return cpu_nodes_.size(); }

  /**
   * @brief Returns the number of gpu ops in the graph.
   */
  inline int NumGPUOp() const { return gpu_nodes_.size(); }

  /**
   * @brief Returns the number of internal ops in the graph.
   */
  inline int NumInternalOp() const { return internal_nodes_.size(); }

  inline OpPtr<CPUBackend>& cpu_op(int idx) {
    NDLL_ENFORCE_VALID_INDEX(idx, (Index)cpu_nodes_.size());
    return cpu_nodes_[idx].op;
  }

  inline OpPtr<GPUBackend>& gpu_op(int idx) {
    NDLL_ENFORCE_VALID_INDEX(idx, (Index)gpu_nodes_.size());
    return gpu_nodes_[idx].op;
  }

  inline unique_ptr<internal::InternalOp>& internal_op(int idx) {
    NDLL_ENFORCE_VALID_INDEX(idx, (Index)internal_nodes_.size());
    return internal_nodes_[idx].op;
  }

  /**
   * @brief Returns the graph node with the given index in the graph.
   */
  OpNode& node(NodeID id);
  
  DISABLE_COPY_MOVE_ASSIGN(OpGraph);
private:
  vector<CPUOpNode> cpu_nodes_;
  vector<GPUOpNode> gpu_nodes_;
  vector<InternalOpNode> internal_nodes_;

  // Stores a mapping from NodeIDs to a pair of integers. The
  // first int in the pair is 0, 1, or 2 for cpu, gpu, internal,
  // and the second is the index of the op within the specified
  // vector.
  vector<std::pair<int, int>> id_to_node_map_;
  
  std::map<string, NodeID> tensor_srcs_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_OP_GRAPH_H_
