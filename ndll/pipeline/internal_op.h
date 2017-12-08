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
   * @brief Implemented by derived operators to perform
   * their computation.
   */
  virtual void Run(MixedWorkspace *ws) = 0;
  
protected:
};

#define USE_INTERNAL_OP_MEMBERS()               \
  using Operator<CPUBackend>::num_threads_;     \
  using Operator<CPUBackend>::batch_size_

NDLL_DECLARE_OPTYPE_REGISTRY(InternalOp, InternalOp);

#define NDLL_REGISTER_INTERNAL_OP(OpName, OpType)         \
  int OPERATOR_SCHEMA_REQUIRED_FOR_##OpName();            \
  static int ANONYMIZE_VARIABLE(OpName) =                 \
    OPERATOR_SCHEMA_REQUIRED_FOR_##OpName();              \
  NDLL_DEFINE_OPTYPE_REGISTERER(OpName, OpType,           \
      InternalOp, InternalOp)

} // namespace internal
} // namespace ndll

#endif // NDLL_PIPELINE_INTERNAL_OP_H_
