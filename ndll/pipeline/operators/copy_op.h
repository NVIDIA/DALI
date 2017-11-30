#ifndef NDLL_PIPELINE_OPERATORS_COPY_OP_H_
#define NDLL_PIPELINE_OPERATORS_COPY_OP_H_

#include <cuda_runtime_api.h>

#include <cstring>

#include "ndll/pipeline/operator.h"

namespace ndll {

template <typename Backend>
class CopyOp : public Operator<Backend> {
public:
  inline explicit CopyOp(const OpSpec &spec) :
    Operator<Backend>(spec) {}
  
  virtual inline ~CopyOp() = default;

  // The CopyOp copies a single input directly to the output
  inline int MaxNumInput() const override { return 1; }
  inline int MinNumInput() const override { return 1; }
  inline int MaxNumOutput() const override { return 1; }
  inline int MinNumOutput() const override { return 1; }
  
  DISABLE_COPY_MOVE_ASSIGN(CopyOp);
protected:
  inline void RunPerSampleCPU(SampleWorkspace *ws) override {
    auto &input = ws->Input<CPUBackend>(0);
    auto output = ws->Output<CPUBackend>(0);
    output->set_type(input.type());
    output->ResizeLike(input);
    std::memcpy(output->raw_mutable_data(),
        input.raw_data(), input.nbytes());
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

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_COPY_OP_H_
