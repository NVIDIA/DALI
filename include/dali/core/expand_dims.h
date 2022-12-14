// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_CORE_EXPAND_DIMS_H_
#define DALI_CORE_EXPAND_DIMS_H_

#include <stdexcept>
#include "dali/core/format.h"
#include "dali/core/tensor_shape.h"

namespace dali {

/**
 * @brief  Inserts new dimensions with extent 1 to the new shape as needed to match a given number
 of dimensions

 *
 * @param sh input shape
 * @param new_dims_position Position where the new dimensions are inserted
 * @param dynamic_out_dim Dynamic number of dimensions in the output (relevant if static_out_ndim ==
 DynamicDimensions)
 */
template <int static_out_ndim = DynamicDimensions, int static_in_ndim = DynamicDimensions>
void expand_dims(TensorListShape<static_out_ndim> &out_sh,
                 const TensorListShape<static_in_ndim> &sh,
                 int new_dims_position = 0,
                 int dynamic_out_dim = -1) {
  int out_ndim = static_out_ndim >= 0 ? static_out_ndim : dynamic_out_dim;
  if (out_ndim < 0)
    throw std::invalid_argument(make_string("Invalid new number of dimensions: ", out_ndim));

  int ndim = sh.sample_dim();
  if (ndim > out_ndim)
    throw std::logic_error(
        make_string("Input has more dimensions than requested: ", ndim, " > ", out_ndim));

  int nsamples = sh.num_samples();
  out_sh.resize(nsamples, out_ndim);
  int insert_ndim = out_ndim - ndim;

  if (new_dims_position < 0 || new_dims_position > ndim) {
    throw std::invalid_argument(make_string("Invalid position: ", new_dims_position,
                                            ". Supported range is [0, ", ndim, "]"));
  }

  for (int s = 0; s < nsamples; s++) {
    auto sample_sh = sh.tensor_shape_span(s);
    auto new_sample_sh = out_sh.tensor_shape_span(s);

    int out_d = 0;
    for (int d = 0; d < ndim; d++) {
      if (d == new_dims_position)
        out_d += insert_ndim;
      new_sample_sh[out_d++] = sample_sh[d];
    }
    for (int out_d = new_dims_position; out_d < new_dims_position + insert_ndim; out_d++) {
      new_sample_sh[out_d] = 1;
    }
  }
}

template <int static_out_ndim = DynamicDimensions, int static_in_ndim = DynamicDimensions>
TensorListShape<static_out_ndim> expand_dims(const TensorListShape<static_in_ndim> &sh,
                                             int new_dims_position = 0, int dynamic_out_dim = -1) {
  TensorListShape<static_out_ndim> out_sh;
  expand_dims(out_sh, sh, new_dims_position, dynamic_out_dim);
  return out_sh;
}

template <int static_out_ndim = DynamicDimensions, int static_in_ndim = DynamicDimensions>
void expand_dims(TensorShape<static_out_ndim> &out_sh, const TensorShape<static_in_ndim> &sh,
                 int new_dims_position = 0, int dynamic_out_dim = -1) {
  int out_ndim = static_out_ndim >= 0 ? static_out_ndim : dynamic_out_dim;
  if (out_ndim < 0)
    throw std::invalid_argument(make_string("Invalid new number of dimensions: ", out_ndim));

  int ndim = sh.sample_dim();
  if (ndim > out_ndim)
    throw std::logic_error(
        make_string("Input has more dimensions than requested: ", ndim, " > ", out_ndim));

  out_sh.resize(out_ndim);
  int insert_ndim = out_ndim - ndim;

  if (new_dims_position < 0 || new_dims_position > ndim) {
    throw std::invalid_argument(make_string("Invalid position: ", new_dims_position,
                                            ". Supported range is [0, ", ndim, "]"));
  }

  int out_d = 0;
  for (int d = 0; d < ndim; d++) {
    if (d == new_dims_position)
      out_d += insert_ndim;
    out_sh[out_d++] = sh[d];
  }
  for (int out_d = new_dims_position; out_d < new_dims_position + insert_ndim; out_d++) {
    out_sh[out_d] = 1;
  }
}

template <int static_out_ndim = DynamicDimensions, int static_in_ndim = DynamicDimensions>
TensorShape<static_out_ndim> expand_dims(const TensorShape<static_in_ndim> &sh,
                                         int new_dims_position = 0, int dynamic_out_dim = -1) {
  TensorShape<static_out_ndim> out_sh;
  expand_dims(out_sh, sh, new_dims_position, dynamic_out_dim);
  return out_sh;
}

}  // namespace dali

#endif  // DALI_CORE_EXPAND_DIMS_H_
