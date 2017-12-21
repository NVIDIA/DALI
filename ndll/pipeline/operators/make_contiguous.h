// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_MAKE_CONTIGUOUS_H_
#define NDLL_PIPELINE_OPERATORS_MAKE_CONTIGUOUS_H_

#include <vector>

#include "ndll/pipeline/internal_op.h"

namespace ndll {
namespace internal {

class MakeContiguous : public InternalOp {
 public:
  inline explicit MakeContiguous(const OpSpec &spec) :
    InternalOp(spec) {}

  virtual inline ~MakeContiguous() = default;

  inline void Run(MixedWorkspace *ws) override {
    vector<Dims> output_shape(batch_size_);
    TypeInfo type = ws->Input<CPUBackend>(0, 0).type();
    for (int i = 0; i < batch_size_; ++i) {
      auto &input = ws->Input<CPUBackend>(0, i);
      output_shape[i] = input.shape();
      NDLL_ENFORCE(type == input.type(), "Inconsistent types in "
          "input batch. Cannot copy to contiguous device buffer.");
    }

    if (ws->OutputIsType<CPUBackend>(0)) {
      auto output = ws->Output<CPUBackend>(0);
      output->Resize(output_shape);
      output->set_type(type);

      for (int i = 0; i < batch_size_; ++i) {
        auto &input = ws->Input<CPUBackend>(0, i);

        // Note: We know that this will translate into
        // a std::memcpy, so it is safe to pass stream 0
        type.Copy<CPUBackend, CPUBackend>(
            output->raw_mutable_tensor(i),
            input.raw_data(), input.size(), 0);
      }
    } else {
      auto output = ws->Output<GPUBackend>(0);
      output->Resize(output_shape);
      output->set_type(type);

      for (int i = 0; i < batch_size_; ++i) {
        auto &input = ws->Input<CPUBackend>(0, i);
        CUDA_CALL(cudaMemcpyAsync(
                output->raw_mutable_tensor(i),
                input.raw_data(),
                input.nbytes(),
                cudaMemcpyHostToDevice,
                ws->stream()));
      }
    }
  }

  DISABLE_COPY_MOVE_ASSIGN(MakeContiguous);

 protected:
  USE_INTERNAL_OP_MEMBERS();
};

}  // namespace internal
}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_MAKE_CONTIGUOUS_H_
