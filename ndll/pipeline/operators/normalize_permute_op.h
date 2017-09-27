#ifndef NDLL_PIPELINE_OPERATORS_NORMALIZE_PERMUTE_H_
#define NDLL_PIPELINE_OPERATORS_NORMALIZE_PERMUTE_H_

#include "ndll/image/transform.h"
#include "ndll/pipeline/data/backend.h"
#include "ndll/pipeline/data/tensor.h"
#include "ndll/pipeline/operators/operator.h"

namespace ndll {

template <typename Backend, typename OUT>
class NormalizePermuteOp : public Transformer<Backend> {
public:
  inline NormalizePermuteOp(vector<float> mean, vector<float> std, int H, int W, int C)
    : mean_vec_(mean), std_vec_(std), H_(H), W_(W), C_(C) {
    NDLL_ENFORCE(H_ > 0);
    NDLL_ENFORCE(W_ > 0);
    NDLL_ENFORCE(C_ == 3 || C_ == 1);
    NDLL_ENFORCE((int)mean.size() == C_);
    NDLL_ENFORCE((int)std.size() == C_);

    // Inverse the std-deviation
    vector<float> inv_std(C_);
    for (int i = 0; i < C_; ++i) {
      inv_std[i] = 1.f / std[i];
    }
    
    mean_.Copy(mean);
    inv_std_.Copy(inv_std);
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
  
  inline void SetOutputType(Batch<Backend> *output, TypeMeta input_type) {
    NDLL_ENFORCE(IsType<uint8>(input_type));
    output->template data<OUT>();
  }
  
  inline NormalizePermuteOp* Clone() const override {
    return new NormalizePermuteOp(mean_vec_, std_vec_, H_, W_, C_);
  }

  inline string name() const override {
    return "NormalizePermuteOp";
  }
protected:
  inline void RunBatchedGPU(const Batch<Backend> &input,
      Batch<Backend> *output) override {
    BatchedNormalizePermute(input.template data<uint8>(), batch_size_, H_, W_, C_,
        mean_.template data<float>(), inv_std_.template data<float>(),
        output->template data<OUT>(), stream_pool_->GetStream());
  }
  
  Tensor<Backend> mean_, inv_std_;
  vector<float> mean_vec_, std_vec_;
  int H_, W_, C_;
  
  using Operator<Backend>::num_threads_;
  using Operator<Backend>::batch_size_;
  using Operator<Backend>::stream_pool_;
};

} // namespace ndll

#endif // NDLL_PIPELINE_OPERATORS_NORMALIZE_PERMUTE_H_
