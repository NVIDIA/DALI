// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_INTERNAL_OP_H_
#define NDLL_PIPELINE_INTERNAL_OP_H_

#include "ndll/pipeline/mixed_workspace.h"
#include "ndll/pipeline/operator.h"

namespace ndll {

class InternalOp : public Operator {
 public:
  inline explicit InternalOp(const OpSpec &spec) :
    Operator(spec) {}

  virtual inline ~InternalOp() = default;

};

#define USE_INTERNAL_OP_MEMBERS()               \
  using Operator::num_threads_;     \
  using Operator::batch_size_

NDLL_DECLARE_OPTYPE_REGISTRY(InternalOp, InternalOp);

#define NDLL_REGISTER_INTERNAL_OP(OpName, OpType)         \
  int NDLL_OPERATOR_SCHEMA_REQUIRED_FOR_##OpName();            \
  static int ANONYMIZE_VARIABLE(OpName) =                 \
    NDLL_OPERATOR_SCHEMA_REQUIRED_FOR_##OpName();              \
  NDLL_DEFINE_OPTYPE_REGISTERER(OpName, OpType,           \
      InternalOp, InternalOp)

}  // namespace ndll

#endif  // NDLL_PIPELINE_INTERNAL_OP_H_
