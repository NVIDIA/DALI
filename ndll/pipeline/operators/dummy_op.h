// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_DUMMY_OP_H_
#define NDLL_PIPELINE_OPERATORS_DUMMY_OP_H_

#include "ndll/pipeline/operator.h"

namespace ndll {

template <typename Backend>
class DummyOp : public Operator {
 public:
  inline explicit DummyOp(const OpSpec &spec) :
    Operator(spec) {}

  virtual inline ~DummyOp() = default;

  DISABLE_COPY_MOVE_ASSIGN(DummyOp);

 protected:
  inline void RunPerSampleCPU(SampleWorkspace *, const int) override {
    NDLL_FAIL("I'm a dummy op don't run me");
  }

  inline void RunBatchedGPU(DeviceWorkspace *, const int) override {
    NDLL_FAIL("I'm a dummy op don't run me");
  }
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_DUMMY_OP_H_
