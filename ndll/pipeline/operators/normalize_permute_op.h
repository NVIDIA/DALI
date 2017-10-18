#ifndef NDLL_PIPELINE_OPERATORS_NORMALIZE_PERMUTE_H_
#define NDLL_PIPELINE_OPERATORS_NORMALIZE_PERMUTE_H_

#include "ndll/image/transform.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/transformer.h"

namespace ndll {

template <typename Backend>
class NormalizePermuteOp : public Transformer<Backend> {
public:
  inline NormalizePermuteOp(const OpSpec &spec) :
    Transformer<Backend>(spec),
    output_type_(spec.GetSingleArgument<NDLLDataType>("output_type", NDLL_FLOAT)),
    H_(spec.GetSingleArgument<int>("height", -1)),
    W_(spec.GetSingleArgument<int>("width", -1)),
    C_(spec.GetSingleArgument<int>("channels", -1)) {
    NDLL_ENFORCE(H_ > 0);
    NDLL_ENFORCE(W_ > 0);
    NDLL_ENFORCE(C_ == 3 || C_ == 1);
    
    vector<float> mean = spec.GetRepeatedArgument<float>("mean");
    vector<float> std = spec.GetRepeatedArgument<float>("std");
    NDLL_ENFORCE((int)mean.size() == C_);
    NDLL_ENFORCE((int)std.size() == C_);

    // Inverse the std-deviation
    for (int i = 0; i < C_; ++i) {
      std[i] = 1.f / std[i];
    }

    // Copy to device
    mean_.Copy(mean);
    inv_std_.Copy(std);
  }
    
  virtual inline ~NormalizePermuteOp() = default;
  
  inline vector<Index> InferOutputShapeFromShape(
      const vector<Index> &input_shape, int /* unused */, int /* unused */) override {    
    // Outputs images in CHW layout
    NDLL_ENFORCE(input_shape.size() == 3);
    NDLL_ENFORCE(input_shape[0] == H_);
    NDLL_ENFORCE(input_shape[1] == W_);
    NDLL_ENFORCE(input_shape[2] == C_);
        
    return {C_, input_shape[0], input_shape[1]};
  }
  
  inline void SetOutputType(Batch<Backend> *output, TypeInfo input_type) {
    NDLL_ENFORCE(IsType<uint8>(input_type));
    if (output_type_ == NDLL_FLOAT) {
      output->template mutable_data<float>();
    } else if (output_type_ == NDLL_FLOAT16) {
      output->template mutable_data<float16>();
    } else {
      NDLL_FAIL("Unsupported output type.");
    }
  }
  
  inline string name() const override {
    return "NormalizePermuteOp";
  }
protected:
  inline void RunBatchedGPU(const Batch<Backend> &input,
      Batch<Backend> *output) override {
    if (output_type_ == NDLL_FLOAT) {
      RunHelper<float>(input, output);
    } else if (output_type_ == NDLL_FLOAT16) {
      RunHelper<float16>(input, output);
    } else {
      NDLL_FAIL("Unsupported output type.");
    }
  }

  template <typename OUT>
  inline void RunHelper(const Batch<Backend> &input, Batch<Backend> *output) {
    NDLL_CALL(BatchedNormalizePermute(input.template data<uint8>(), batch_size_, H_, W_, C_,
            mean_.template mutable_data<float>(), inv_std_.template mutable_data<float>(),
            output->template mutable_data<OUT>(), stream_));
  }
  
  Tensor<Backend> mean_, inv_std_;
  NDLLDataType output_type_;
  int H_, W_, C_;
  
  using Operator<Backend>::num_threads_;
  using Operator<Backend>::batch_size_;
  using Operator<Backend>::stream_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_NORMALIZE_PERMUTE_H_
