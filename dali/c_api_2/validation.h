// Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_C_API_2_VALIDATION_H_
#define DALI_C_API_2_VALIDATION_H_

#include <stdexcept>
#include <optional>
#define DALI_ALLOW_NEW_C_API
#include "dali/dali.h"
#include "dali/core/format.h"
#include "dali/core/span.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/pipeline/data/types.h"

namespace dali::c_api {

inline void Validate(daliDataType_t dtype) {
  if (!TypeTable::TryGetTypeInfo(dtype))
    throw std::invalid_argument(make_string("Invalid data type: ", dtype));
}

inline void Validate(const TensorLayout &layout, int ndim, bool allow_empty = true) {
  if (layout.empty() && allow_empty)
    return;
  if (layout.ndim() != ndim)
    throw std::invalid_argument(make_string(
      "The layout '", layout, "' cannot describe ", ndim, "-dimensional data."));
}

template <typename ShapeLike>
void ValidateSampleShape(
      int sample_index,
      ShapeLike &&sample_shape,
      std::optional<int> expected_ndim = std::nullopt) {
  int ndim = std::size(sample_shape);
  if (expected_ndim.has_value() && ndim != *expected_ndim)
    throw std::invalid_argument(make_string(
      "Unexpected number of dimensions (", ndim, ") in sample ", sample_index,
      ". Expected ", *expected_ndim, "."));

  for (int j = 0; j < ndim; j++)
    if (sample_shape[j] < 0)
      throw std::invalid_argument(make_string(
        "Negative extent encountered in the shape of sample ", sample_index, ". Offending shape: ",
        TensorShape<-1>(sample_shape)));
}

inline void ValidateNumSamples(int num_samples) {
  if (num_samples < 0)
    throw std::invalid_argument("The number of samples must not be negative.");
}

inline void ValidateNDim(int ndim) {
  if (ndim < 0)
    throw std::invalid_argument("The number of dimensions must not be negative.");
}


inline void ValidateShape(
      int ndim,
      const int64_t *shape) {
  ValidateNDim(ndim);
  if (ndim > 0 && !shape)
    throw std::invalid_argument("The `shape` must not be NULL when ndim > 0.");

  for (int j = 0; j < ndim; j++)
    if (shape[j] < 0)
      throw std::invalid_argument(make_string(
        "The tensor shape must not contain negative extents. Got: ",
        TensorShape<-1>(make_cspan(shape, ndim))));
}

inline void ValidateShape(int num_samples, int ndim, const int64_t *shapes) {
  ValidateNumSamples(num_samples);
  ValidateNDim(ndim);
  if (!shapes && num_samples > 0 && ndim > 0)
    throw std::invalid_argument("The `shapes` are required for non-scalar (ndim>=0) samples.");

  if (ndim > 0) {
    for (int i = 0; i < num_samples; i++)
      ValidateSampleShape(i, make_cspan(&shapes[i*ndim], ndim));
  }
}

inline void Validate(daliStorageDevice_t device_type) {
  if (device_type != DALI_STORAGE_CPU && device_type != DALI_STORAGE_GPU)
    throw std::invalid_argument(make_string("Invalid storage device type: ", device_type));
}

void ValidateDeviceId(int device_id, bool allow_cpu_only);

inline void Validate(const daliBufferPlacement_t &placement) {
  Validate(placement.device_type);
  if (placement.device_type == DALI_STORAGE_GPU || placement.pinned)
    ValidateDeviceId(placement.device_id, placement.pinned);
}

}  // namespace dali::c_api

#endif  // DALI_C_API_2_VALIDATION_H_
