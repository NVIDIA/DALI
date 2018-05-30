// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_UTIL_DUMMY_OP_H_
#define NDLL_PIPELINE_OPERATORS_UTIL_DUMMY_OP_H_

#include "ndll/pipeline/operators/operator.h"

namespace ndll {

template <typename Backend>
class DummyOp : public Operator<Backend> {
 public:
  inline explicit DummyOp(const OpSpec &spec) :
    Operator<Backend>(spec) {}

  virtual inline ~DummyOp() = default;

  DISABLE_COPY_MOVE_ASSIGN(DummyOp);

 protected:
  void RunImpl(Workspace<Backend> *, const int) override {
    NDLL_FAIL("I'm a dummy op don't run me");
  }
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_UTIL_DUMMY_OP_H_
