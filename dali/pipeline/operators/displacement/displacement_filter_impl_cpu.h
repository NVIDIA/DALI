// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef DALI_PIPELINE_OPERATORS_DISPLACEMENT_DISPLACEMENT_FILTER_IMPL_CPU_H_
#define DALI_PIPELINE_OPERATORS_DISPLACEMENT_DISPLACEMENT_FILTER_IMPL_CPU_H_

#include "dali/common.h"
#include "dali/pipeline/operators/displacement/displacement_filter.h"

namespace dali {

template <class Displacement,
          bool per_channel_transform>
class DisplacementFilter<CPUBackend, Displacement,
                         per_channel_transform> : public Operator<CPUBackend> {
 public:
  explicit DisplacementFilter(const OpSpec &spec) :
      Operator(spec),
      displace_(spec),
      interp_type_(spec.GetArgument<DALIInterpType>("interp_type")) {
    has_mask_ = spec.HasTensorArgument("mask");
    param_.set_pinned(false);
    DALI_ENFORCE(interp_type_ == DALI_INTERP_NN || interp_type_ == DALI_INTERP_LINEAR,
        "Unsupported interpolation type, only NN and LINEAR are supported for this operation");
    try {
      fill_value_ = spec.GetArgument<float>("fill_value");
    } catch (std::runtime_error e) {
      try {
        fill_value_ = spec.GetArgument<int>("fill_value");
      } catch (std::runtime_error e) {
        DALI_FAIL("Invalid type of argument \"fill_value\". Expected int or float");
      }
    }
  }

  virtual ~DisplacementFilter() {
    displace_.Cleanup();
  }

  void RunImpl(SampleWorkspace* ws, const int idx) override {
    DataDependentSetup(ws, idx);

    auto &input = ws->Input<CPUBackend>(idx);
    switch (interp_type_) {
      case DALI_INTERP_NN:
        if (IsType<float>(input.type())) {
          PerSampleCPULoop<float, DALI_INTERP_NN>(ws, idx);
        } else if (IsType<uint8_t>(input.type())) {
          PerSampleCPULoop<uint8_t, DALI_INTERP_NN>(ws, idx);
        } else {
          DALI_FAIL("Unexpected input type " + input.type().name());
        }
        break;
      case DALI_INTERP_LINEAR:
        if (IsType<float>(input.type())) {
          PerSampleCPULoop<float, DALI_INTERP_LINEAR>(ws, idx);
        } else if (IsType<uint8_t>(input.type())) {
          PerSampleCPULoop<uint8_t, DALI_INTERP_LINEAR>(ws, idx);
        } else {
          DALI_FAIL("Unexpected input type " + input.type().name());
        }
        break;
      default:
        DALI_FAIL("Unsupported interpolation type,"
            " only NN and LINEAR are supported for this operation");
    }
  }

  template <typename U = Displacement>
  typename std::enable_if<HasParam<U>::value>::type PrepareDisplacement(SampleWorkspace *ws) {
    param_.Resize({1});
    param_.mutable_data<typename U::Param>();

    typename U::Param &p = param_.mutable_data<typename U::Param>()[0];
    displace_.Prepare(&p, spec_, ws, ws->data_idx());
    displace_.param = p;
  }

  template <typename U = Displacement>
  typename std::enable_if<!HasParam<U>::value>::type PrepareDisplacement(SampleWorkspace *) {}

  /**
   * @brief Do basic input checking and output setup
   * assuming output_shape = input_shape
   */
  virtual void DataDependentSetup(SampleWorkspace *ws, const int idx) {
    auto &input = ws->Input<CPUBackend>(idx);
    auto *output = ws->Output<CPUBackend>(idx);
    output->ResizeLike(input);
  }

  void SetupSharedSampleParams(SampleWorkspace *ws) override {
    if (has_mask_) {
      mask_ = &(ws->ArgumentInput("mask"));
    }
    PrepareDisplacement(ws);
  }

  USE_OPERATOR_MEMBERS();

 private:
  template <typename T, DALIInterpType interp_type>
  bool PerSampleCPULoop(SampleWorkspace *ws, const int idx) {
    auto& input = ws->Input<CPUBackend>(idx);
    auto *output = ws->Output<CPUBackend>(idx);

    const auto H = input.shape()[0];
    const auto W = input.shape()[1];
    const auto C = input.shape()[2];

    auto *in = input.data<T>();
    auto *out = output->template mutable_data<T>();

    if (!has_mask_ || mask_->template data<bool>()[ws->data_idx()]) {
      for (Index h = 0; h < H; ++h) {
        for (Index w = 0; w < W; ++w) {
          // calculate displacement for all channels at once
          // vs. per-channel
          if (per_channel_transform) {
            for (Index c = 0; c < C; ++c) {
              // output idx is set by location
              Index out_idx = (h * W + w) * C + c;
              // input idx is calculated by function
              T out_value;
              if (interp_type == DALI_INTERP_NN) {
                // NN interpolation
                Point<Index> p = displace_.template operator()<Index>(h, w, c, H, W, C);
                if (p.x >= 0 && p.y >= 0) {
                  Index in_idx = (p.y * W + p.x) * C + c;
                  out_value = in[in_idx];
                } else {
                  out_value = fill_value_;
                }
              } else {
                // LINEAR interpolation
                Point<float> p = displace_.template operator()<float>(h, w, c, H, W, C);
                if (p.x >= 0 && p.y >= 0) {
                  T inter_values[4];
                  Index x = p.x;
                  Index y = p.y;
                  Index xp = x < W - 1 ? x + 1 : x;
                  Index yp = y < H - 1 ? y + 1 : y;
                  // 0, 0
                  inter_values[0] = in[(y * W + x) * C + c];
                  // 1, 0
                  inter_values[1] = in[(y * W + xp) * C + c];
                  // 0, 1
                  inter_values[2] = in[(yp * W + x) * C + c];
                  // 1, 1
                  inter_values[3] = in[(yp * W + xp) * C + c];
                  const float rx = p.x - x;
                  const float ry = p.y - y;
                  const float mrx = 1 - rx;
                  const float mry = 1 - ry;
                  out_value = static_cast<T>(
                      inter_values[0] * mrx * mry +
                      inter_values[1] * rx * mry +
                      inter_values[2] * mrx * ry +
                      inter_values[3] * rx * ry);
                } else {
                  out_value = fill_value_;
                }
              }

              // copy
              out[out_idx] = out_value;
            }
          } else {
            // output idx is set by location
            Index out_idx = (h * W + w) * C;
            if (interp_type == DALI_INTERP_NN) {
              // input idx is calculated by function
              Point<Index> p = displace_.template operator()<Index>(h, w, 0, H, W, C);
              if (p.x >= 0 && p.y >= 0) {
                Index in_idx = (p.y * W + p.x) * C;

                // apply transform uniformly across channels
                for (int c = 0; c < C; ++c) {
                  out[out_idx+c] = in[in_idx + c];
                }
              } else {
                for (int c = 0; c < C; ++c) {
                  out[out_idx+c] = fill_value_;
                }
              }
            } else {
              Point<float> p = displace_.template operator()<float>(h, w, 0, H, W, C);
              if (p.x >= 0 && p.y >= 0) {
                T inter_values[4];
                Index x = p.x;
                Index y = p.y;
                Index xp = x < W - 1 ? x + 1 : x;
                Index yp = y < H - 1 ? y + 1 : y;
                const float rx = p.x - x;
                const float ry = p.y - y;
                const float mrx = 1 - rx;
                const float mry = 1 - ry;
                for (int c = 0; c < C; ++c) {
                  // 0, 0
                  inter_values[0] = in[(y * W + x) * C + c];
                  // 1, 0
                  inter_values[1] = in[(y * W + xp) * C + c];
                  // 0, 1
                  inter_values[2] = in[(yp * W + x) * C + c];
                  // 1, 1
                  inter_values[3] = in[(yp * W + xp) * C + c];
                  out[out_idx + c] = static_cast<T>(
                      inter_values[0] * mrx * mry +
                      inter_values[1] * rx * mry +
                      inter_values[2] * mrx * ry +
                      inter_values[3] * rx * ry);
                }
              } else {
                for (int c = 0; c < C; ++c) {
                  out[out_idx+c] = fill_value_;
                }
              }
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

  Displacement displace_;
  DALIInterpType interp_type_;
  float fill_value_;

  bool has_mask_;
  const Tensor<CPUBackend> * mask_;

  Tensor<CPUBackend> param_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DISPLACEMENT_DISPLACEMENT_FILTER_IMPL_CPU_H_
