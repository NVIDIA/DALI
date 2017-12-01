#ifndef NDLL_PIPELINE_OPERATORS_DUMMY_OP_H_
#define NDLL_PIPELINE_OPERATORS_DUMMY_OP_H_

#include "ndll/pipeline/operator.h"

namespace ndll {

template <typename Backend>
class DummyOp : public Operator<Backend> {
public:
  inline explicit DummyOp(const OpSpec &spec) :
    Operator<Backend>(spec) {}
  
  virtual inline ~DummyOp() = default;

  // The DummyOp copies a single input directly to the output
  inline int MaxNumInput() const override { return 10; }
  inline int MinNumInput() const override { return 0; }
  inline int MaxNumOutput() const override { return 10; }
  inline int MinNumOutput() const override { return 0; }
  
  DISABLE_COPY_MOVE_ASSIGN(DummyOp);
protected:
  inline void RunPerSampleCPU(SampleWorkspace *ws) override {
    NDLL_FAIL("I'm a dummy op don't run me");
  }
  
  inline void RunBatchedGPU(DeviceWorkspace *ws) override {
    NDLL_FAIL("I'm a dummy op don't run me");
  }
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_DUMMY_OP_H_
