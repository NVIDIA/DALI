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
  
  inline vector<Index> InferOutputShapeFromShape(
      const vector<Index> &input_shape, int /* unused */, int /* unused */) override {
    return input_shape;
  }
  
  inline void SetOutputType(Batch<Backend> *output, TypeInfo input_type) {
    output->set_type(input_type);
  }
  
  inline string name() const override {
    return "CopyOp";
  }
protected:
  inline void RunPerDatumCPU(const Datum<Backend> &input,
      Datum<Backend> *output, int /* unused */, int /* unused */) override {
    NDLL_ENFORCE(input.shape() == output->shape());
    std::memcpy(output->raw_mutable_data(), input.raw_data(), input.nbytes());
  }
  
  inline void RunBatchedGPU(const Batch<Backend> &input,
      Batch<Backend> *output) override {
    CUDA_CALL(cudaMemcpyAsync(
            output->raw_mutable_data(),
            input.raw_data(),
            input.nbytes(),
            cudaMemcpyDeviceToDevice,
            stream_));
  }
  
  using Operator<Backend>::num_threads_;
  using Operator<Backend>::stream_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_COPY_OP_H_
