#ifndef NDLL_PIPELINE_OP_GRAPH_H_
#define NDLL_PIPELINE_OP_GRAPH_H_

#include <map>
#include <unordered_set>

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
  std::unordered_set<NodeID> parents, children;
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

// Stores meta-data about a tensor and how it
// is used by a producer/consumer node.
struct TensorMeta {
  NodeID node;
  int index;
  bool is_cpu;
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
   * @brief Removes the node with the specified NodeID from 
   * the graph. Fails if the removal would produce an invalid
   * graph.
   */
  void RemoveOp(NodeID id);
  
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

  /**
   * @brief Returns a reference to the `idx`-th cpu op that was
   * added to the graph.
   */
  inline Operator<CPUBackend>& cpu_op(int idx) {
    NDLL_ENFORCE_VALID_INDEX(idx, (Index)cpu_nodes_.size());
    return *cpu_nodes_[idx].op;
  }

  /**
   * @brief Returns the node object for the `idx`-th cpu op that
   * was added to the graph.
   */
  inline CPUOpNode& cpu_node(int idx) {
    NDLL_ENFORCE_VALID_INDEX(idx, (Index)cpu_nodes_.size());
    return cpu_nodes_[idx];
  }
  
  /**
   * @brief Returns a reference to the `idx`-th gpu op that
   * was added to the graph.
   */
  inline Operator<GPUBackend>& gpu_op(int idx) {
    NDLL_ENFORCE_VALID_INDEX(idx, (Index)gpu_nodes_.size());
    return *gpu_nodes_[idx].op;
  }

  /**
   * @brief Returns the node object for the `idx`-th gpu op that
   * was added to the graph.
   */
  inline GPUOpNode& gpu_node(int idx) {
    NDLL_ENFORCE_VALID_INDEX(idx, (Index)gpu_nodes_.size());
    return gpu_nodes_[idx];
  }
  
  /**
   * @brief Returns a reference to the `idx`-th internal op
   * that was added to the graph.
   */
  inline internal::InternalOp& internal_op(int idx) {
    NDLL_ENFORCE_VALID_INDEX(idx, (Index)internal_nodes_.size());
    return *internal_nodes_[idx].op;
  }

  /**
   * @brief Returns the node object for the `idx`-th internal op that
   * was added to the graph.
   */
  inline InternalOpNode& internal_node(int idx) {
    NDLL_ENFORCE_VALID_INDEX(idx, (Index)internal_nodes_.size());
    return internal_nodes_[idx];
  }
  
  /**
   * @brief Returns the graph node with the given index in the graph.
   */
  OpNode& node(NodeID id);

  /**
   * @brief Returns the type (cpu, gpu, internal) of the node
   * at the given index.
   */
  inline NDLLOpType NodeType(NodeID id) const {
    NDLL_ENFORCE_VALID_INDEX(id, (Index)id_to_node_map_.size());
    return id_to_node_map_[id].first;
  }

  /**
   * @brief Returns the index of the node with the specified id
   * among nodes of its type.
   */
  inline int NodeIdx(NodeID id) const {
    NDLL_ENFORCE_VALID_INDEX(id, (Index)id_to_node_map_.size());
    return id_to_node_map_[id].second;
  }

  /**
   * @brief Returns the TensorMeta objects for the tensor 
   * with the given name and its producer node.
   */
  inline TensorMeta TensorSourceMeta(const string &name) const {
    auto it = tensor_producers_.find(name);
    NDLL_ENFORCE(it != tensor_producers_.end(), "Tensor with name \"" +
        name + "\" has no know source.");
    return it->second;
  }
  
  /**
   * @brief Returns the id of the op that produces the tensor with
   * the given name.
   */
  inline NodeID TensorSourceID(const string &name) {
    return TensorSourceMeta(name).node;
  }

  /**
   * @brief Returns the output idx of the input tensor in 
   * its source.
   */
  inline int TensorIdxInSource(const string &name) {
    return TensorSourceMeta(name).index;
  }
  
  /**
   * @brief Returns true if the tensor with the given name
   * has a backend type that matches the calling type.
   */
  template <typename Backend>
  bool TensorIsType(const string &name);

  /**
   * @brief Returns a vector of meta-data about the nodes that
   * consume the tensor with the input name.
   */
  inline vector<TensorMeta> TensorConsumerMeta(const string &name) const {
    auto it = tensor_consumers_.find(name);
    if (it == tensor_consumers_.end()) {
      // If we have no entries for this tensors consumers,
      // we just return an empty vector
      return vector<TensorMeta>{};
    }
    return it->second;
  }
  
  DISABLE_COPY_MOVE_ASSIGN(OpGraph);
private:
  vector<CPUOpNode> cpu_nodes_;
  vector<GPUOpNode> gpu_nodes_;
  vector<InternalOpNode> internal_nodes_;

  // Stores a mapping from NodeIDs to a pair where the first
  // element indicates what type of node it is,  and the second
  // is the index of the op within the specified vector.
  vector<std::pair<NDLLOpType, int>> id_to_node_map_;

  std::map<string, TensorMeta> tensor_producers_;
  std::map<string, vector<TensorMeta>> tensor_consumers_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_OP_GRAPH_H_
