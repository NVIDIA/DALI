// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#ifndef NDLL_PIPELINE_OPERATORS_DISPLACEMENT_FILTER_IMPL_CPU_H_
#define NDLL_PIPELINE_OPERATORS_DISPLACEMENT_FILTER_IMPL_CPU_H_

#include <random>

#include "ndll/common.h"
#include "ndll/pipeline/operator.h"

namespace ndll {

template <class Displacement,
          class Augment,
          bool per_channel_transform>
class DisplacementFilter<CPUBackend, Displacement,
                         Augment, per_channel_transform> : public Operator<CPUBackend> {
 public:
  explicit DisplacementFilter(const OpSpec &spec) :
    Operator(spec),
    displace_(spec),
    augment_(spec),
    rand_gen_(spec.GetArgument<int>("seed")),
    dis(spec.GetArgument<float>("probability")) {
      mask_.set_pinned(false);
  }

  virtual ~DisplacementFilter() {
    displace_.Cleanup();
    augment_.Cleanup();
  }

  void RunImpl(SampleWorkspace* ws, const int idx) override {
    DataDependentSetup(ws, idx);

    auto &input = ws->Input<CPUBackend>(idx);
    if (IsType<float>(input.type())) {
      PerSampleCPULoop<float>(ws, idx);
    } else if (IsType<uint8_t>(input.type())) {
      PerSampleCPULoop<uint8_t>(ws, idx);
    } else {
      NDLL_FAIL("Unexpected input type " + input.type().name());
    }
  }

  /**
   * @brief Do basic input checking and output setup
   * assuming output_shape = input_shape
   */
  virtual void DataDependentSetup(SampleWorkspace *ws, const int idx) {
    auto &input = ws->Input<CPUBackend>(idx);
    auto *output = ws->Output<CPUBackend>(idx);
    output->ResizeLike(input);
  }

 private:
  template <typename T>
  bool PerSampleCPULoop(SampleWorkspace *ws, const int idx) {
    auto& input = ws->Input<CPUBackend>(idx);
    auto *output = ws->Output<CPUBackend>(idx);

    const auto H = input.shape()[0];
    const auto W = input.shape()[1];
    const auto C = input.shape()[2];

    auto *in = input.data<T>();
    auto *out = output->template mutable_data<T>();

    if (mask_.template data<bool>()[idx]) {
      for (Index h = 0; h < H; ++h) {
        for (Index w = 0; w < W; ++w) {
          // calculate displacement for all channels at once
          // vs. per-channel
          if (per_channel_transform) {
            for (Index c = 0; c < C; ++c) {
              // output idx is set by location
              Index out_idx = (h * W + w) * C + c;
              // input idx is calculated by function
              Index in_idx = displace_(h, w, c, H, W, C);

              // copy
              out[out_idx] = augment_(in[in_idx], h, w, c, H, W, C);
            }
          } else {
            // output idx is set by location
            Index out_idx = (h * W + w) * C;
            // input idx is calculated by function
            Index in_idx = displace_(h, w, 0, H, W, C);

            // apply transform uniformly across channels
            for (int c = 0; c < C; ++c) {
              out[out_idx+c] = augment_(in[in_idx + c], h, w, c, H, W, C);
            }
          }
        }
      }
    } else {  // Do not do augmentation, pass through
      for (Index h = 0; h < H; ++h) {
        for (Index w = 0; w < W; ++w) {
          for (int c = 0; c < C; ++c) {
            out[(h * W + w) * C + c] = in[(h * W + w) * C + c];
          }
        }
      }
    }
    return true;
  }

  void SetupSharedSampleParams(SampleWorkspace *ws) override {
    mask_.Resize({batch_size_});
    mask_.mutable_data<bool>();

    for (int i = 0; i < batch_size_; ++i) {
      mask_.template mutable_data<bool>()[i] = dis(rand_gen_);
    }
  }

  USE_OPERATOR_MEMBERS();

 private:
  Displacement displace_;
  Augment augment_;

  std::mt19937 rand_gen_;
  Tensor<CPUBackend> mask_;
  std::bernoulli_distribution dis;
};

}  // namespace ndll

#endif  // NDLL_PIPELINE_OPERATORS_DISPLACEMENT_FILTER_IMPL_CPU_H_
