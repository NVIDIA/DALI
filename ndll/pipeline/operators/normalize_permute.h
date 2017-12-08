#ifndef NDLL_PIPELINE_OPERATORS_NORMALIZE_PERMUTE_H_
#define NDLL_PIPELINE_OPERATORS_NORMALIZE_PERMUTE_H_

#include "ndll/image/transform.h"
#include "ndll/pipeline/operator.h"

namespace ndll {

template <typename Backend>
class NormalizePermute : public Operator<Backend> {
public:
  inline NormalizePermute(const OpSpec &spec) :
    Operator<Backend>(spec),
    output_type_(spec.GetArgument<NDLLDataType>("output_type", NDLL_FLOAT)),
    H_(spec.GetArgument<int>("height", -1)),
    W_(spec.GetArgument<int>("width", -1)),
    C_(spec.GetArgument<int>("channels", -1)) {
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

    // TODO(tgale): We don't really want to do this in
    // the default stream, we should make at least some
    // stream available for constructors of ops.
    mean_.Copy(mean, 0);
    inv_std_.Copy(std, 0);

    output_shape_.resize(batch_size_);
    for (auto &shape : output_shape_) shape = {C_, H_, W_};
  }
    
  virtual inline ~NormalizePermute() = default;
  
protected:
  inline void RunBatchedGPU(DeviceWorkspace *ws) override {
    if (output_type_ == NDLL_FLOAT) {
      RunHelper<float>(ws);
    } else if (output_type_ == NDLL_FLOAT16) {
      RunHelper<float16>(ws);
    } else {
      NDLL_FAIL("Unsupported output type.");
    }
  }

  template <typename OUT>
  inline void RunHelper(DeviceWorkspace *ws) {
    auto &input = ws->Input<GPUBackend>(0);
    auto output = ws->Output<GPUBackend>(0);
    
    // Validate input shape and type
    NDLL_ENFORCE(IsType<uint8>(input.type()));
    NDLL_ENFORCE(input.ntensor() == batch_size_,
        "Input does not have batch_size samples.");
    for (int i = 0; i < batch_size_; ++i) {
      NDLL_ENFORCE(input.tensor_shape(i).size() == 3,
          "Expects 3-dim image input.");
      NDLL_ENFORCE(input.tensor_shape(i)[0] == H_,
          "Input image height does not match output height.");
      NDLL_ENFORCE(input.tensor_shape(i)[1] == W_,
          "Input image width does not match output width.");
      NDLL_ENFORCE(input.tensor_shape(i)[2] == C_,
          "Input image channels does not match output channels.");
    }
    
    // Resize the output & run
    output->Resize(output_shape_);
    NDLL_CALL(BatchedNormalizePermute(
            input.template data<uint8>(),
            batch_size_, H_, W_, C_,
            mean_.template mutable_data<float>(),
            inv_std_.template mutable_data<float>(),
            output->template mutable_data<OUT>(),
            ws->stream()));
  }
  
  Tensor<Backend> mean_, inv_std_;
  NDLLDataType output_type_;
  int H_, W_, C_;
  vector<Dims> output_shape_;

  USE_OPERATOR_MEMBERS();
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_NORMALIZE_PERMUTE_H_
