// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_COPY_H_
#define NDLL_PIPELINE_OPERATORS_COPY_H_

#include <cuda_runtime_api.h>

#include <cstring>

#include "ndll/pipeline/operator.h"

namespace ndll {

template <typename Backend>
class Copy : public Operator<Backend> {
public:
  inline explicit Copy(const OpSpec &spec) :
    Operator<Backend>(spec) {}
  
  virtual inline ~Copy() = default;
  
  DISABLE_COPY_MOVE_ASSIGN(Copy);
protected:
  inline void RunPerSampleCPU(SampleWorkspace *ws) override {
    auto &input = ws->Input<CPUBackend>(0);
    auto output = ws->Output<CPUBackend>(0);
    output->set_type(input.type());
    output->ResizeLike(input);

    TypeInfo type = input.type();
    type.Copy<CPUBackend, CPUBackend>(
        output->raw_mutable_data(),
        input.raw_data(), input.size(), 0);
  }

  inline void RunBatchedGPU(DeviceWorkspace *ws) override {
    auto &input = ws->Input<GPUBackend>(0);
    auto output = ws->Output<GPUBackend>(0);
    output->set_type(input.type());
    output->ResizeLike(input);
    CUDA_CALL(cudaMemcpyAsync(
            output->raw_mutable_data(),
            input.raw_data(),
            input.nbytes(),
            cudaMemcpyDeviceToDevice,
            ws->stream()));
  }
};

}  // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_COPY_H_
