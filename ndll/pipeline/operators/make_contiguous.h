// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_MAKE_CONTIGUOUS_H_
#define NDLL_PIPELINE_OPERATORS_MAKE_CONTIGUOUS_H_

#include <vector>

#include "ndll/pipeline/operator.h"
#include "ndll/common.h"

// Found by benchmarking coalesced vs non coalesced on diff size images
#define COALESCE_TRESHOLD 8192

namespace ndll {

class MakeContiguous : public Operator<Mixed> {
 public:
  inline explicit MakeContiguous(const OpSpec &spec) :
    Operator<Mixed>(spec),
    coalesced(true)
    {}

  virtual inline ~MakeContiguous() = default;

  void Run(MixedWorkspace *ws) override {
    vector<Dims> output_shape(batch_size_);
    TypeInfo type = ws->Input<CPUBackend>(0, 0).type();
    for (int i = 0; i < batch_size_; ++i) {
      auto &input = ws->Input<CPUBackend>(0, i);
      output_shape[i] = input.shape();
      if (coalesced && input.nbytes() > COALESCE_TRESHOLD)
        coalesced = false;
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

      if (coalesced) {
        TimeRange tm("coalesced", TimeRange::kBlue);
        cpu_output_buff.ResizeLike(*output);
        cpu_output_buff.set_type(type);
        for (int i = 0; i < batch_size_; ++i) {
          auto &input = ws->Input<CPUBackend>(0, i);
          memcpy(cpu_output_buff.raw_mutable_tensor(i), input.raw_data(), input.nbytes());
        }
        CUDA_CALL(cudaMemcpyAsync(
              output->raw_mutable_data(),
              cpu_output_buff.raw_mutable_data(),
              cpu_output_buff.nbytes(),
              cudaMemcpyHostToDevice,
              ws->stream()));
      } else {
        TimeRange tm("non coalesced", TimeRange::kGreen);
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
    coalesced = true;
  }

  DISABLE_COPY_MOVE_ASSIGN(MakeContiguous);

 protected:
  USE_OPERATOR_MEMBERS();
  TensorList<CPUBackend> cpu_output_buff;
  bool coalesced;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_MAKE_CONTIGUOUS_H_
