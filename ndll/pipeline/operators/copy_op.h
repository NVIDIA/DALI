#ifndef NDLL_PIPELINE_OPERATORS_COPY_OP_H_
#define NDLL_PIPELINE_OPERATORS_COPY_OP_H_

#include <cstring>

#include "ndll/pipeline/transformer.h"

namespace ndll {

template <typename Backend>
class CopyOp : public Transformer<Backend> {
public:
  inline explicit CopyOp(const OpSpec &spec) :
    Transformer<Backend>(spec) {}
  
  virtual inline ~CopyOp() = default;
  
  inline vector<Dims> InferOutputShapesFromShapes(const SampleMeta &meta) override {
    return meta.InputShapes();
  }
  
  inline vector<TypeInfo> InferOutputTypes(const BatchMeta &meta) override {
    return meta.InputTypes();
  }
  
  inline string name() const override {
    return "CopyOp";
  }
  
protected:
  inline void RunPerSampleCPU(SampleWorkspace *ws) override {
    auto input = ws->Input<GPUBackend>(0);
    auto output = ws->Output<GPUBackend>(0);
    std::memcpy(output->raw_mutable_data(),
        input.raw_data(), input.nbytes());
  }
  
  inline void RunBatchedGPU(BatchWorkspace *ws) override {
    auto input = ws->Input<GPUBackend>(0);
    auto output = ws->Output<GPUBackend>(0);
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
