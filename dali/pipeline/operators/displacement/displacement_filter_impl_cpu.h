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

#include <array>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/pipeline/operators/displacement/displacement_filter.h"

namespace dali {

template <class Displacement, bool per_channel_transform>
class DisplacementFilter<CPUBackend, Displacement, per_channel_transform>
    : public Operator<CPUBackend> {
 public:
  explicit DisplacementFilter(const OpSpec &spec)
      : Operator(spec),
        displace_(num_threads_, Displacement(spec)),
        interp_type_(spec.GetArgument<DALIInterpType>("interp_type")) {
    has_mask_ = spec.HasTensorArgument("mask");
    DALI_ENFORCE(
        interp_type_ == DALI_INTERP_NN || interp_type_ == DALI_INTERP_LINEAR,
        "Unsupported interpolation type, only NN and LINEAR are supported for "
        "this operation");
    if (!spec.TryGetArgument<float>(fill_value_, "fill_value")) {
      int int_value = 0;
      if (!spec.TryGetArgument<int>(int_value, "fill_value")) {
        DALI_FAIL("Invalid type of argument \"fill_value\". Expected int or float");
      }
      fill_value_ = int_value;
    }
  }

  ~DisplacementFilter() override {
    for (auto &d : displace_) {
      d.Cleanup();
    }
  }

  void RunImpl(SampleWorkspace *ws, const int idx) override {
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
        DALI_FAIL(
            "Unsupported interpolation type,"
            " only NN and LINEAR are supported for this operation");
    }
  }

  template <typename U = Displacement>
  typename std::enable_if<HasParam<U>::value>::type PrepareDisplacement(
      SampleWorkspace *ws) {
    auto *p = &displace_[ws->thread_idx()].param;
    displace_[ws->thread_idx()].Prepare(p, spec_, ws, ws->data_idx());
  }

  template <typename U = Displacement>
  typename std::enable_if<!HasParam<U>::value>::type PrepareDisplacement(
      SampleWorkspace *) {}

  /**
   * @brief Do basic input checking and output setup
   * assuming output_shape = input_shape
   */
  virtual void DataDependentSetup(SampleWorkspace *ws, const int idx) {
    auto &input = ws->Input<CPUBackend>(idx);
    auto &output = ws->Output<CPUBackend>(idx);
    output.ResizeLike(input);
  }

  void SetupSharedSampleParams(SampleWorkspace *ws) override {
    if (has_mask_) {
      mask_ = &(ws->ArgumentInput("mask"));
    }
    PrepareDisplacement(ws);
  }

  USE_OPERATOR_MEMBERS();
  using Operator<CPUBackend>::RunImpl;

 private:
  // TODO(klecki) We could probably interpolate with something other than float,
  // for example integer type
  struct LinearCoefs {
    LinearCoefs(float rx, float ry)
        : rx(rx), ry(ry), mrx(1.f - rx), mry(1.f - ry) {}
    LinearCoefs() = delete;
    const float rx, ry, mrx, mry;
  };

  template <typename T>
  bool ShouldTransform(Point<T> p) {
    return p.x >= 0 && p.y >= 0;
  }

  template <typename T>
  Index PointToInIdx(Point<T> p, Index H, Index W, Index C) {
    Point<Index> p_idx = p.template Cast<Index>();
    return (p_idx.y * W + p_idx.x) * C;
  }

  LinearCoefs PointToLinearCoefs(Point<float> p) {
    Point<Index> p_idx = p.template Cast<Index>();
    return {p.x - p_idx.x, p.y - p_idx.y};
  }

  std::pair<Index, LinearCoefs> PointToInIdxCoefs(Point<float> p, Index H,
                                                  Index W, Index C) {
    Point<Index> p_idx = p.template Cast<Index>();
    return {PointToInIdx(p_idx, H, W, C),
            LinearCoefs{p.x - p_idx.x, p.y - p_idx.y}};
  }

  // Calculate offset to next element on x and y axis with border handling
  template <typename T>
  Point<Index> CalcNextOffsets(Point<T> p, Index H, Index W, Index C) {
    Point<Index> p_idx = p.template Cast<Index>();
    return {p_idx.x < W - 1 ? C : 0, p_idx.y < H - 1 ? C * W : 0};
  }

  template <typename T>
  std::array<T, 4> load_inputs(const T *in, Index in_idx,
                               Point<Index> next_offsets) {
    std::array<T, 4> result;
    // 0, 0
    result[0] = in[in_idx];
    // 1, 0
    result[1] = in[in_idx + next_offsets.x];
    // 0, 1
    result[2] = in[in_idx + next_offsets.y];
    // 1, 1
    result[3] = in[in_idx + next_offsets.x + next_offsets.y];
    return result;
  }

  template <typename T>
  T linear_interpolate(std::array<T, 4> inter_values, LinearCoefs coefs) {
    return static_cast<T>(inter_values[0] * coefs.mrx * coefs.mry +
                          inter_values[1] * coefs.rx * coefs.mry +
                          inter_values[2] * coefs.mrx * coefs.ry +
                          inter_values[3] * coefs.rx * coefs.ry);
  }

  template <typename T, DALIInterpType interp_type>
  bool PerSampleCPULoop(SampleWorkspace *ws, const int idx) {
    auto &input = ws->Input<CPUBackend>(idx);
    auto &output = ws->Output<CPUBackend>(idx);

    DALI_ENFORCE(input.ndim() == 3, "Operator expects 3-dimensional image input.");

    const auto H = input.shape()[0];
    const auto W = input.shape()[1];
    const auto C = input.shape()[2];

    auto *in = input.data<T>();
    auto *out = output.template mutable_data<T>();

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
                const auto p =
                    displace_[ws->thread_idx()].template operator()<Index>(
                        h, w, c, H, W, C);
                if (ShouldTransform(p)) {
                  const auto in_idx = PointToInIdx(p, H, W, C) + c;
                  out_value = in[in_idx];
                } else {
                  out_value = fill_value_;
                }
              } else {
                // LINEAR interpolation
                const auto p =
                    displace_[ws->thread_idx()].template operator()<float>(
                        h, w, c, H, W, C);
                if (ShouldTransform(p)) {
                  const auto in_idx = PointToInIdx(p, H, W, C) + c;
                  const auto next_offsets = CalcNextOffsets(p, H, W, C);
                  const auto linear_coefs = PointToLinearCoefs(p);
                  const auto inter_values =
                      load_inputs(in, in_idx, next_offsets);
                  out_value = linear_interpolate(inter_values, linear_coefs);
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
              const auto p =
                  displace_[ws->thread_idx()].template operator()<Index>(
                      h, w, 0, H, W, C);
              if (ShouldTransform(p)) {
                const auto in_idx = PointToInIdx(p, H, W, C);
                // apply transform uniformly across channels
                for (int c = 0; c < C; ++c) {
                  out[out_idx + c] = in[in_idx + c];
                }
              } else {
                for (int c = 0; c < C; ++c) {
                  out[out_idx + c] = fill_value_;
                }
              }
            } else {
              const auto p =
                  displace_[ws->thread_idx()].template operator()<float>(
                      h, w, 0, H, W, C);
              if (ShouldTransform(p)) {
                const auto in_idx = PointToInIdx(p, H, W, C);
                const auto next_offsets = CalcNextOffsets(p, H, W, C);
                const auto linear_coefs = PointToLinearCoefs(p);
                for (int c = 0; c < C; ++c) {
                  const auto inter_values =
                      load_inputs(in, in_idx + c, next_offsets);
                  out[out_idx + c] =
                      linear_interpolate(inter_values, linear_coefs);
                }
              } else {
                for (int c = 0; c < C; ++c) {
                  out[out_idx + c] = fill_value_;
                }
              }
            }
          }
        }
      }
    } else {  // Do not do augmentation, pass through
      // TODO(klecki) just use memcpy
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

  std::vector<Displacement> displace_;
  DALIInterpType interp_type_;
  float fill_value_;

  bool has_mask_;
  const Tensor<CPUBackend> *mask_;
};

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATORS_DISPLACEMENT_DISPLACEMENT_FILTER_IMPL_CPU_H_
