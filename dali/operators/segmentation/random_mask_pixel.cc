// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include <random>
#include <utility>
#include "dali/core/static_switch.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/batch_rng.h"
#include "dali/operators/segmentation/utils/searchable_rle_mask.h"
#include "dali/kernels/common/utils.h"
#include "dali/core/boundary.h"

#define MASK_SUPPORTED_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, \
                              uint64_t, int64_t, float, bool)

namespace dali {

DALI_SCHEMA(segmentation__RandomMaskPixel)
    .DocStr(R"(Selects random pixel coordinates in a mask, sampled from a uniform distribution.

Based on run-time argument ``foreground``, it returns either only foreground pixels or any pixels.

Pixels are classificed as foreground either when their value exceeds a given ``threshold`` or when
it's equal to a specific ``value``.
)")
    .AddOptionalArg<int>("value",
      R"code(All pixels equal to this value are interpreted as foreground.

This argument is mutually exclusive with ``threshold`` argument and is meant to be used only
with integer inputs.
)code", nullptr, true)
    .AddOptionalArg<float>("threshold",
      R"code(All pixels with a value above this threshold are interpreted as foreground.

This argument is mutually exclusive with ``value`` argument.
)code", 0.0f, true)
    .AddOptionalArg("foreground",
      R"code(If different than 0, the pixel position is sampled uniformly from all foreground pixels.

If 0, the pixel position is sampled uniformly from all available pixels.)code",
      0, true)
    .NumInput(1)
    .NumOutput(1);

class RandomMaskPixelCPU : public Operator<CPUBackend> {
 public:
  explicit RandomMaskPixelCPU(const OpSpec &spec);
  bool CanInferOutputs() const override { return true; }
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) override;
  void RunImpl(workspace_t<CPUBackend> &ws) override;

 private:
  template <typename T>
  void RunImplTyped(workspace_t<CPUBackend> &ws);

  BatchRNG<std::mt19937> rngs_;
  std::vector<SearchableRLEMask> rle_;

  std::vector<int> foreground_;
  std::vector<int> value_;
  std::vector<float> threshold_;

  bool has_value_ = false;

  USE_OPERATOR_MEMBERS();
};

RandomMaskPixelCPU::RandomMaskPixelCPU(const OpSpec &spec)
    : Operator<CPUBackend>(spec),
      rngs_(spec.GetArgument<int64_t>("seed"), spec.GetArgument<int64_t>("max_batch_size")),
      has_value_(spec.ArgumentDefined("value")) {
  if (has_value_) {
    DALI_ENFORCE(!spec.ArgumentDefined("threshold"),
                 "Arguments ``value`` and ``threshold`` can not be provided together");
  }
}

bool RandomMaskPixelCPU::SetupImpl(std::vector<OutputDesc> &output_desc,
                                    const workspace_t<CPUBackend> &ws) {
  const auto &in_masks = ws.template InputRef<CPUBackend>(0);
  int nsamples = in_masks.size();
  auto in_masks_shape = in_masks.shape();
  int ndim = in_masks_shape.sample_dim();
  output_desc.resize(1);
  output_desc[0].shape = uniform_list_shape(nsamples, {ndim});
  output_desc[0].type = TypeTable::GetTypeInfo(DALI_INT64);

  foreground_.resize(nsamples);
  value_.clear();
  threshold_.clear();

  GetPerSampleArgument(foreground_, "foreground", ws, nsamples);
  if (spec_.ArgumentDefined("value")) {
    GetPerSampleArgument(value_, "value", ws, nsamples);
  } else {
    GetPerSampleArgument(threshold_, "threshold", ws, nsamples);
  }
  return true;
}

template <typename T>
void RandomMaskPixelCPU::RunImplTyped(workspace_t<CPUBackend> &ws) {
  const auto &in_masks = ws.template InputRef<CPUBackend>(0);
  auto &out_pixel_pos = ws.template OutputRef<CPUBackend>(0);
  int nsamples = in_masks.size();
  auto in_masks_shape = in_masks.shape();
  int ndim = in_masks_shape.sample_dim();
  auto masks_view = view<const T>(in_masks);
  auto pixel_pos_view = view<int64_t>(out_pixel_pos);
  auto& thread_pool = ws.GetThreadPool();

  rle_.resize(thread_pool.NumThreads());

  for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
    thread_pool.AddWork(
      [&, sample_idx](int thread_id) {
        auto &rng = rngs_[sample_idx];
        auto mask = masks_view[sample_idx];
        auto pixel_pos = pixel_pos_view[sample_idx];
        const auto &mask_sh = mask.shape;
        if (foreground_[sample_idx]) {
          int64_t flat_idx = -1;
          auto &rle_mask = rle_[thread_id];
          rle_mask.Clear();
          if (has_value_) {
            T value = static_cast<T>(value_[sample_idx]);
            // checking if the value is representable by T, otherwise we
            // just fall back to pick a random pixel.
            if (static_cast<int>(value) == value_[sample_idx]) {
              rle_mask.Init(
                  mask, [value](const T &x) { return x == value; });
            }
          } else {
            float threshold = threshold_[sample_idx];
            rle_mask.Init(
                mask, [threshold](const T &x) { return x > threshold; });
          }
          if (rle_mask.count() > 0) {
            auto dist = std::uniform_int_distribution<int64_t>(0, rle_mask.count() - 1);
            flat_idx = rle_mask.find(dist(rng));
          }
          if (flat_idx >= 0) {
            // Convert from flat_idx to per-dim indices
            auto mask_strides = kernels::GetStrides(mask_sh);
            for (int d = 0; d < ndim - 1; d++) {
              pixel_pos.data[d] = flat_idx / mask_strides[d];
              flat_idx = flat_idx % mask_strides[d];
            }
            pixel_pos.data[ndim - 1] = flat_idx;
            return;
          }
        }
        // Either foreground == 0 or no foreground pixels found. Get a random center
        for (int d = 0; d < ndim; d++) {
          pixel_pos.data[d] = std::uniform_int_distribution<int64_t>(0, mask_sh[d] - 1)(rng);
        }
      }, in_masks_shape.tensor_size(sample_idx));
  }
  thread_pool.RunAll();
}

void RandomMaskPixelCPU::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &in_masks = ws.template InputRef<CPUBackend>(0);
  TYPE_SWITCH(in_masks.type().id(), type2id, T, MASK_SUPPORTED_TYPES, (
    RunImplTyped<T>(ws);
  ), (  // NOLINT
    DALI_FAIL(make_string("Unexpected data type: ", in_masks.type().id()));
  ));  // NOLINT
}

DALI_REGISTER_OPERATOR(segmentation__RandomMaskPixel, RandomMaskPixelCPU, CPU);

}  // namespace dali
