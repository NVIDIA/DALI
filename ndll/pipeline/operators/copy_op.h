#ifndef NDLL_PIPELINE_OPERATORS_COPY_OP_H_
#define NDLL_PIPELINE_OPERATORS_COPY_OP_H_

#include "ndll/pipeline/operators/operator.h"

namespace ndll {

template <typename Backend>
class CopyOp : public Transformer<Backend> {
public:
  inline CopyOp() {}
  virtual inline ~CopyOp() = default;

  inline void RunBatchedGPU(const Batch<Backend> &input,
      Batch<Backend> *output) override {
    CUDA_ENFORCE(cudaMemcpyAsync(
            output->raw_data(),
            input.raw_data(),
            input.nbytes(),
            cudaMemcpyDeviceToDevice,
            stream_pool_->GetStream()));
  }
  
  inline vector<Index> InferOutputShapeFromShape(
      const vector<Index> &input_shape, int /* unused */) override {
    return input_shape;
  }
  
  inline void SetOutputType(Batch<Backend> *output, TypeMeta input_type) {
    output->set_type(input_type);
  }
  
  inline CopyOp* Clone() const override {
    return new CopyOp;
  }

  inline string name() const override {
    return "CopyOp";
  }
protected:
  using Operator<Backend>::num_threads_;
  using Operator<Backend>::stream_pool_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_COPY_OP_H_
