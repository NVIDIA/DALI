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
#include "dali/operators/segmentation/utils/searchable_rle_mask.h"
#include "dali/kernels/common/utils.h"
#include "dali/core/boundary.h"

#define MASK_SUPPORTED_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, \
                              uint64_t, int64_t, float)

namespace dali {

DALI_SCHEMA(segmentation__BiasedCropCenter)
    .DocStr(R"(Selects a cropping window center which can be selected randomly either
at any position in the input or at the position of any nonzero pixel in the input,
based on an ``nonzero`` argument, typically the result of a coin flip operation.

The purpose of this operator is to obtain a distribution of cropping window centers that
has a certain bias towards selecting a nonzero pixel as a center.

The output of the operator can be used for cropping the images::

    import nvidia.dali.fn as fn
    shape = (height, width)
    center = fn.biased_crop_center(mask, shape=shape)
    anchor = center - shape // 2
    cropped = fn.slice(image, anchor, shape, axis_names="HW")

..note::

    Since the centers are selected uniformly from all the available nonzero pixels, larger
objects in the mask can be oversampled.
)")
    .AddOptionalArg("nonzero",
      R"code(If different than 0, the crop center is selected to match any of the nonzero
 pixels in the input mask. If 0, the crop center is selected randomly.)code",
      0, true)
    .AddOptionalArg<std::vector<int>>("shape",
      R"code(If specified, it represents the shape of the cropping window to be used,
introducing restrictions to the range of valid crop centers.

When nonzero == 0, a random cropping center is selected so that the cropping window
is fully contained in the input.

When nonzero != 0, the cropping center is first picked to match any of the nonzero
pixels in the input. If the selected crop center results in an out of bounds cropping window,
the center is shifted as necessary so that the window remains within bounds.

If left unspecified, the shape is not taken into account when selecting the center,
effectively resulting in any position in the input being a valid center.
)code",
      nullptr, true)
    .NumInput(1)
    .NumOutput(1);

class BiasedCropCenterCPU : public Operator<CPUBackend> {
 public:
  explicit BiasedCropCenterCPU(const OpSpec &spec);
  bool CanInferOutputs() const override { return true; }
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<CPUBackend> &ws) override;
  void RunImpl(workspace_t<CPUBackend> &ws) override;

 private:
  template <typename T>
  void RunImplTyped(workspace_t<CPUBackend> &ws);

  int64_t seed_;
  std::vector<std::mt19937_64> rng_;

  std::vector<int> nonzero_;
  TensorListShape<> crop_shape_;

  bool has_crop_shape_ = false;

  USE_OPERATOR_MEMBERS();
};

BiasedCropCenterCPU::BiasedCropCenterCPU(const OpSpec &spec)
    : Operator<CPUBackend>(spec), seed_(spec.GetArgument<int64_t>("seed")) {
}

bool BiasedCropCenterCPU::SetupImpl(std::vector<OutputDesc> &output_desc,
                                    const workspace_t<CPUBackend> &ws) {
  const auto &in_masks = ws.template InputRef<CPUBackend>(0);
  int nsamples = in_masks.size();
  auto in_masks_shape = in_masks.shape();
  int ndim = in_masks_shape.sample_dim();
  output_desc.resize(1);
  output_desc[0].shape = uniform_list_shape(nsamples, {ndim});
  output_desc[0].type = TypeTable::GetTypeInfo(DALI_INT64);

  nonzero_.clear();
  if (spec_.HasTensorArgument("nonzero")) {
    nonzero_.resize(nsamples);
    const auto &nonzero_arg_in = ws.ArgumentInput("nonzero");
    for (int i = 0; i < nsamples; i++) {
      nonzero_[i] = *nonzero_arg_in[i].data<int>();
    }
  } else {
    nonzero_.resize(nsamples, spec_.GetArgument<int>("nonzero"));
  }

  has_crop_shape_ = spec_.ArgumentDefined("shape");
  if (has_crop_shape_) {
    GetShapeArgument(crop_shape_, spec_, "shape", ws, ndim, nsamples);
  }
  return true;
}

template <typename T>
void BiasedCropCenterCPU::RunImplTyped(workspace_t<CPUBackend> &ws) {
  const auto &in_masks = ws.template InputRef<CPUBackend>(0);
  auto &out_center = ws.template OutputRef<CPUBackend>(0);
  int nsamples = in_masks.size();
  auto in_masks_shape = in_masks.shape();
  int ndim = in_masks_shape.sample_dim();
  auto masks_view = view<const T>(in_masks);
  auto center_view = view<int64_t>(out_center);
  auto& thread_pool = ws.GetThreadPool();

  if (rng_.empty()) {
    for (int i = 0; i < thread_pool.size(); i++) {
      rng_.emplace_back(seed_ + i);
    }
  }
  assert(rng_.size() == static_cast<size_t>(thread_pool.size()));

  for (int sample_idx = 0; sample_idx < nsamples; sample_idx++) {
    thread_pool.AddWork(
      [&, sample_idx](int thread_id) {
        auto &rng = rng_[thread_id];
        auto mask = masks_view[sample_idx];
        auto center = center_view[sample_idx];
        const auto &mask_sh = mask.shape;
        auto crop_sh = crop_shape_.tensor_shape_span(sample_idx);
        if (nonzero_[sample_idx]) {
          SearchableRLEMask rle_mask(mask);
          if (rle_mask.count() > 0) {
            auto dist = std::uniform_int_distribution<int64_t>(0, rle_mask.count() - 1);
            int64_t flat_idx = rle_mask.find(dist(rng));

            // Convert from flat_idx to per-dim indices
            auto mask_strides = kernels::GetStrides(mask_sh);
            for (int d = 0; d < ndim - 1; d++) {
              center.data[d] = flat_idx / mask_strides[d];
              flat_idx = flat_idx % mask_strides[d];
            }
            center.data[ndim - 1] = flat_idx;

            // Adjust center if necessary
            if (has_crop_shape_) {
              for (int d = 0; d < ndim; d++) {
                int64_t half = crop_sh[d] >> 1;
                center.data[d] = clamp(center.data[d], half, mask_sh[d] - (crop_sh[d] - half));
              }
            }
            return;
          }
        }
        // Either nonzero == 0 or no nonzero pixels found. Get a random center
        if (has_crop_shape_) {
          for (int d = 0; d < ndim; d++) {
            int64_t halfl = crop_sh[d] >> 1;
            int64_t start = halfl;
            int64_t end = mask_sh[d] - (crop_sh[d] - halfl);
            center.data[d] = std::uniform_int_distribution<int64_t>(start, end - 1)(rng);
          }
        } else {
          for (int d = 0; d < ndim; d++) {
            center.data[d] = std::uniform_int_distribution<int64_t>(0, mask_sh[d] - 1)(rng);
          }
        }
      }, in_masks_shape.tensor_size(sample_idx));
  }
  thread_pool.RunAll();
}

void BiasedCropCenterCPU::RunImpl(workspace_t<CPUBackend> &ws) {
  const auto &in_masks = ws.template InputRef<CPUBackend>(0);
  TYPE_SWITCH(in_masks.type().id(), type2id, T, MASK_SUPPORTED_TYPES, (
    RunImplTyped<T>(ws);
  ), (  // NOLINT
    DALI_FAIL(make_string("Unexpected data type: ", in_masks.type().id()));
  ));  // NOLINT
}

DALI_REGISTER_OPERATOR(segmentation__BiasedCropCenter, BiasedCropCenterCPU, CPU);

}  // namespace dali
