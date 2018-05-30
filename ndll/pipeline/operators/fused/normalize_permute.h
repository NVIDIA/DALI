// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_FUSED_NORMALIZE_PERMUTE_H_
#define NDLL_PIPELINE_OPERATORS_FUSED_NORMALIZE_PERMUTE_H_

#include <vector>

#include "ndll/pipeline/operators/operator.h"

namespace ndll {

template <typename Backend>
class NormalizePermute : public Operator<Backend> {
 public:
  explicit inline NormalizePermute(const OpSpec &spec) :
    Operator<Backend>(spec),
    output_type_(spec.GetArgument<NDLLDataType>("output_dtype")),
    H_(spec.GetArgument<int>("height")),
    W_(spec.GetArgument<int>("width")),
    C_(spec.GetArgument<int>("channels")) {
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
  void RunImpl(Workspace<Backend> *ws, const int idx) override;

  template <typename OUT>
  void CPURunHelper(const Tensor<CPUBackend> &input, Tensor<CPUBackend> *output);

  template <typename OUT>
  void GPURunHelper(DeviceWorkspace *ws, const int idx);

  Tensor<Backend> mean_, inv_std_;
  NDLLDataType output_type_;
  int H_, W_, C_;
  vector<Dims> output_shape_;

  USE_OPERATOR_MEMBERS();
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_FUSED_NORMALIZE_PERMUTE_H_
