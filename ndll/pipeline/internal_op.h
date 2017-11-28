#ifndef NDLL_PIPELINE_INTERNAL_OP_H_
#define NDLL_PIPELINE_INTERNAL_OP_H_

#include "ndll/pipeline/mixed_workspace.h"
#include "ndll/pipeline/operator.h"

namespace ndll {
namespace internal {

class InternalOp : public Operator<CPUBackend> {
public:
  inline explicit InternalOp(const OpSpec &spec) :
    Operator<CPUBackend>(spec) {}

  virtual inline ~InternalOp() = default;

  /**
   * @brief Implemented by derived operators to perform any
   * batch-wise setup e.g. resizing the output TensorList.
   */
  void Setup(MixedWorkspace *ws) = 0;
  
protected:
};

NDLL_DECLARE_OPTYPE_REGISTRY(InternalOp, InternalOp);

#define NDLL_REGISTER_INTERNAL_OP(OpName, OpType) \
  NDLL_DEFINE_OPTYPE_REGISTERER(OpName, OpType,   \
      InternalOp, InternalOp)

} // namespace internal
} // namespace ndll

#endif // NDLL_PIPELINE_INTERNAL_OP_H_
