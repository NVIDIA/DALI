#ifndef NDLL_PIPELINE_OPERATORS_MAKE_CONTIGUOUS_H_
#define NDLL_PIPELINE_OPERATORS_MAKE_CONTIGUOUS_H_

#include "ndll/pipeline/internal_op.h"

namespace ndll {
namespace internal {

class MakeContiguous : public InternalOp {
public:
  inline explicit MakeContiguous(const OpSpec &spec) :
    InternalOp(spec) {}

  virtual inline ~MakeContiguous() = default;

  inline int MaxNumInput() const override { return 1; }
  inline int MinNumInput() const override { return 1; }
  inline int MaxNumOutput() const override { return 1; }
  inline int MinNumOutput() const override { return 1; }

  inline string name() const override {
    return "MakeContiguous";
  }

  inline void Setup(MixedWorkspace *ws) override {
    vector<Dims> output_shape(batch_size_);
    TypeInfo type = ws->Input<CPUBackend>(0, 0).type();
    for (int i = 0; i < batch_size_; ++i) {
      auto &input = ws->Input<CPUBackend>(0, i);
      output_shape[i] = input.shape();
      NDLL_ENFORCE(type == input.type(), "Inconsistent types in "
          "input batch. Cannot copy to contiguous device buffer.");
    }
    
    auto output = ws->Output<CPUBackend>(0);
    output->Resize(output_shape);
    output->set_type(type);
  }
    
  DISABLE_COPY_MOVE_ASSIGN(MakeContiguous);
protected:
  inline void RunPerSampleCPU(SampleWorkspace *ws) override {
    auto &input = ws->Input<CPUBackend>(0);
    auto output = ws->Output<CPUBackend>(0);

    // Note: Stream doesn't matter here
    output->Copy(input, ws->stream());
  }

  USE_INTERNAL_OP_MEMBERS();
};

} // namespace internal
} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_MAKE_CONTIGUOUS_H_
