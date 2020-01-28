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

#ifndef DALI_OPERATORS_UTIL_DIAG_MSG_H_
#define DALI_OPERATORS_UTIL_DIAG_MSG_H_

#include <iostream>
#include <sstream>
#include <string>
#include "dali/core/format.h"
#include "dali/core/tensor_view.h"
#include "dali/core/tensor_shape_print.h"

namespace dali {

/**
 * @brief Prints a human-readable diagnostic for mismatching TensorListShapes
 *
 * The function first checks dimensionality, then number of samples. If these match,
 * individual tensors are compared and first print_max_samples are added to the message.
 * If there are more mismatches, the function adds a note about the number of remaining mismatches.
 *
 * If the shapes match, the function prints nothing.
 */
template <int ndim>
inline void PrintShapeMismatchMsg(
    std::ostream &stream,
    const TensorListShape<ndim> &a,
    const TensorListShape<ndim> &b,
    int print_max_samples = 4) {
  if (a.sample_dim() != b.sample_dim()) {
    print(stream, "Sample dimensionality differs: ", a.sample_dim(), " vs ", b.sample_dim());
    return;
  }

  if (a.num_samples() != b.num_samples()) {
    print(stream, "Number of samples in the lists differs: ",
      a.num_samples(), " vs ", b.num_samples());
    return;
  }

  int nerr = 0;
  for (int i = 0; i < a.num_samples(); i++) {
    if (a[i] != b[i]) {
      nerr++;
      if (nerr <= print_max_samples) {
        print(stream, "Sample #", i, " shape mismatch: ", a[i], " vs ", b[i], "\n");
      }
    }
  }
  if (nerr > print_max_samples)
    print(stream, " ...and ", nerr - print_max_samples, " more mismatches");
}

template <int ndim>
inline std::string ShapeMismatchMsg(
    const TensorListShape<ndim> &a,
    const TensorListShape<ndim> &b,
    int print_max_samples = 4) {
  std::stringstream ss;
  PrintShapeMismatchMsg(ss, a, b, print_max_samples);
  return ss.str();
}

}  // namespace dali

#endif  // DALI_OPERATORS_UTIL_DIAG_MSG_H_
