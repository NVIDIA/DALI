// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#include "ndll/pipeline/operators/normalize_permute.h"

namespace ndll {

  template<>
  void NormalizePermute<CPUBackend>::RunImpl(SampleWorkspace *ws, const int idx) {
    auto &input = ws->Input<CPUBackend>(idx);
    auto output = ws->Output<CPUBackend>(idx);

    NDLL_ENFORCE(IsType<uint8>(input.type()));
    NDLL_ENFORCE(input.ndim() == 3,
        "Expects 3-dim image input.");
    NDLL_ENFORCE(input.dim(0) == H_,
        "Input image height does not match output height.");
    NDLL_ENFORCE(input.dim(1) == W_,
          "Input image width does not match output width.");
    NDLL_ENFORCE(input.dim(2) == C_,
        "Input image channels does not match output channels.");

    // Output is CHW
    output->Resize({C_, H_, W_});
    if (output_type_ == NDLL_FLOAT) {
      CPURunHelper<float>(input, output);
    } else {
      NDLL_FAIL("Unsupported output type.");
    }
  }

template<>
template <typename OUT>
void NormalizePermute<CPUBackend>::CPURunHelper(const Tensor<CPUBackend> &input, Tensor<CPUBackend> *output) {
  const uint8 *in = input.template data<uint8>();
  OUT *out = output->template mutable_data<OUT>();
  float *mean = mean_.template mutable_data<float>();
  float *inv_std = inv_std_.template mutable_data<float>();

  for (int c = 0; c < C_; ++c) {
    for (int h = 0; h < H_; ++h) {
      for (int w = 0; w < W_; ++w) {
        out[c*H_*W_ + h*W_ + w] = static_cast<OUT>(
            (static_cast<float>(in[h*W_*C_ + w*C_ + c]) - mean[c]) * inv_std[c]);
      }
    }
  }
}

NDLL_REGISTER_OPERATOR(NormalizePermute, NormalizePermute<CPUBackend>, CPU);

NDLL_OPERATOR_SCHEMA(NormalizePermute)
  .DocStr("Foo")
  .NumInput(1)
  .NumOutput(1)
  .AllowMultipleInputSets()
  .AddOptionalArg("output_dtype", "Output data type", NDLL_FLOAT)
  .AddArg("height", "Height of the input image")
  .AddArg("width", "Width of the input image")
  .AddArg("channels", "Number of channels of input image")
  .AddArg("mean", "Mean values of pixels for image normalizations")
  .AddArg("std", "Standard deviation for image normalization");

}  // namespace ndll
