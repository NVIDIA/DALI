// Copyright (c) 2019-2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/operators/image/resize/resampling_attr.h"
#include "dali/pipeline/operator/common.h"

namespace dali {

DALI_SCHEMA(ResamplingFilterAttr)
  .DocStr(R"code(Resampling filter attribute placeholder)code")
  .AddOptionalArg("interp_type",
      R"code(Type of interpolation to be used.

Use ``min_filter`` and ``mag_filter`` to specify different filtering for downscaling and upscaling.
)code",
      DALI_INTERP_LINEAR, true)
  .AddOptionalArg("mag_filter", "Filter used when scaling up.",
      DALI_INTERP_LINEAR, true)
  .AddOptionalArg("min_filter", "Filter used when scaling down.",
      DALI_INTERP_LINEAR, true)
  .AddOptionalArg<DALIDataType>("dtype", R"code(Output data type.

Must be same as input type or ``float``. If not set, input type is used.)code", nullptr)
  .AddOptionalArg("temp_buffer_hint",
      R"code(Initial size in bytes, of a temporary buffer for resampling.

.. note::
  This argument is ignored for the CPU variant.)code",
      0)
  .AddOptionalArg("minibatch_size", R"code(Maximum number of images that are processed in
a kernel call.)code",
      32);


using namespace kernels;  // NOLINT

inline ResamplingFilterType interp2resample(DALIInterpType interp) {
#define DALI_MAP_INTERP_TO_RESAMPLE(interp, resample) case DALI_INTERP_##interp:\
  return ResamplingFilterType::resample;

  switch (interp) {
    DALI_MAP_INTERP_TO_RESAMPLE(NN, Nearest);
    DALI_MAP_INTERP_TO_RESAMPLE(LINEAR, Linear);
    DALI_MAP_INTERP_TO_RESAMPLE(CUBIC, Cubic);
    DALI_MAP_INTERP_TO_RESAMPLE(LANCZOS3, Lanczos3);
    DALI_MAP_INTERP_TO_RESAMPLE(GAUSSIAN, Gaussian);
    DALI_MAP_INTERP_TO_RESAMPLE(TRIANGULAR, Triangular);
  default:
    DALI_FAIL("Unknown interpolation type");
  }
#undef DALI_MAP_INTERP_TO_RESAMPLE
}

void ResamplingFilterAttr::PrepareFilterParams(
      const OpSpec &spec, const ArgumentWorkspace &ws, int num_samples) {
  GetPerSampleArgument(interp_type_arg_, "interp_type", spec, ws, num_samples);
  GetPerSampleArgument(min_arg_, "min_filter", spec, ws, num_samples);
  GetPerSampleArgument(mag_arg_, "mag_filter", spec, ws, num_samples);
  if (!spec.TryGetArgument(dtype_arg_, "dtype"))
    dtype_arg_ = DALI_NO_TYPE;
  bool has_interp = spec.ArgumentDefined("interp_type");
  bool has_min = spec.ArgumentDefined("min_filter");
  bool has_mag = spec.ArgumentDefined("mag_filter");

  min_filter_.resize(num_samples, ResamplingFilterType::Triangular);
  mag_filter_.resize(num_samples, ResamplingFilterType::Linear);

  auto convert = [](auto &filter_types, auto &interp_types) {
    for (int i = 0, n = filter_types.size(); i < n; i++)
      filter_types[i] = interp2resample(interp_types[i]);
  };

  if (has_min)
    convert(min_filter_, min_arg_);
  else if (has_interp)
    convert(min_filter_, interp_type_arg_);

  if (has_mag)
    convert(mag_filter_, mag_arg_);
  else if (has_interp)
    convert(mag_filter_, interp_type_arg_);
}

void ResamplingFilterAttr::GetResamplingParams(
      span<kernels::ResamplingParams> resample_params,
      span<const ResizeParams> resize_params) const {
  for (int i = 0, p = 0; i < resize_params.size(); i++) {
    auto &resz_par = resize_params[i];
    for (int d = 0; d < resz_par.size(); d++) {
      auto &rsmp_par = resample_params[p++];
      rsmp_par.roi = {};
      if (resz_par.src_lo[d] != resz_par.src_hi[d])
        rsmp_par.roi = { resz_par.src_lo[d], resz_par.src_hi[d] };
      rsmp_par.output_size = resz_par.dst_size[d];
      rsmp_par.min_filter = { min_filter_[i], 0 };
      rsmp_par.mag_filter = { mag_filter_[i], 0 };
    }
  }
}

void ResamplingFilterAttr::ApplyFilterParams(
      span<kernels::ResamplingParams> resample_params, int ndim) const {
  int N = resample_params.size() / ndim;
  assert(static_cast<int>(min_filter_.size()) == N);
  assert(static_cast<int>(mag_filter_.size()) == N);
  for (int i = 0, p = 0; i < N; i++) {
    for (int d = 0; d < ndim; d++, p++) {
      resample_params[p].min_filter = { min_filter_[i], 0 };
      resample_params[p].mag_filter = { mag_filter_[i], 0 };
    }
  }
}

}  // namespace dali
