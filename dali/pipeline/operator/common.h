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

#ifndef DALI_PIPELINE_OPERATOR_COMMON_H_
#define DALI_PIPELINE_OPERATOR_COMMON_H_

#include <vector>
#include <string>

#include "dali/core/error_handling.h"
#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/data/views.h"

namespace dali {
template <typename T>
inline void GetSingleOrRepeatedArg(const OpSpec &spec, vector<T> &result,
                                   const std::string &argName, size_t repeat_count = 2) {
  if (!spec.TryGetRepeatedArgument<T>(result, argName)) {
      T scalar = spec.GetArgument<T>(argName);
      result.assign(repeat_count, scalar);
  } else if (result.size() == 1 && repeat_count != 1) {
      T scalar = result.front();
      result.assign(repeat_count, scalar);
  }

  DALI_ENFORCE(result.size() == repeat_count,
      "Argument \"" + argName + "\" expects either a single value "
      "or a list of " + to_string(repeat_count) + " elements. " +
      to_string(result.size()) + " given.");
}

template <typename T>
void GetPerSampleArgument(std::vector<T> &output,
                          const std::string &argument_name,
                          const OpSpec &spec,
                          const ArgumentWorkspace &ws,
                          int batch_size = -1 /* -1 = get from "batch_size" arg */) {
  if (batch_size < 0)
    batch_size = spec.GetArgument<int>("batch_size");

  if (spec.HasTensorArgument(argument_name)) {
    const auto &arg = ws.ArgumentInput(argument_name);
    decltype(auto) shape = arg.shape();
    int N = shape.num_samples();
    if (N == 1) {
      bool is_valid_shape = shape.tensor_shape(0) == TensorShape<1>{batch_size};

      DALI_ENFORCE(is_valid_shape,
        make_string("`", argument_name, "` must be a 1xN or Nx1 (N = ", batch_size,
                    ") tensor list. Got: ", shape));

      output.resize(batch_size);
      auto *data = arg[0].template data<T>();

      for (int i = 0; i < batch_size; i++) {
        output[i] = data[i];
      }
    } else {
      bool is_valid_shape = N == batch_size &&
                            is_uniform(shape) &&
                            volume(shape.tensor_shape_span(0)) == 1;
      DALI_ENFORCE(is_valid_shape,
        make_string("`", argument_name, "` must be a 1xN or Nx1 (N = ", batch_size,
                    ") tensor list. Got: ", shape));

      output.resize(batch_size);
      for (int i = 0; i < batch_size; i++) {
        output[i] = arg[i].template data<T>()[0];
      }
    }
  } else {
    output.clear();
    output.resize(batch_size, spec.GetArgument<T>(argument_name));
  }
  assert(output.size() == static_cast<size_t>(batch_size));
}

template <typename ArgumentType = int, int out_ndim>
void GetShapeArgument(TensorListShape<out_ndim> &out_tls,
                      const OpSpec &spec,
                      const std::string &argument_name,
                      const ArgumentWorkspace &ws,
                      int ndim = out_ndim,
                      int batch_size = -1 /* -1 = get from "batch_size" arg */) {
  if (batch_size < 0)
    batch_size = spec.GetArgument<int>("batch_size");

  auto to_extent = [](ArgumentType extent) {
    return std::is_floating_point<ArgumentType>::value
      ? static_cast<int64_t>(std::floor(extent + 0.5))
      : static_cast<int64_t>(extent);
  };

  // if neither is dynamic, ndim and out_ndim must be equal
  assert(ndim < 0 || out_ndim < 0 || ndim == out_ndim);

  if (spec.HasTensorArgument(argument_name)) {
    const auto &arg = ws.ArgumentInput(argument_name);
    auto argview = view<const ArgumentType>(arg);
    int N = argview.shape.num_samples();
    DALI_ENFORCE(N == batch_size, make_string("Unexpected number of samples in argument `",
      argument_name, "` (expected: ", batch_size, ")"));
    DALI_ENFORCE(is_uniform(argview.shape), "A tensor list shape must have the same dimensionality "
      "for all samples.");
    TensorListShape<out_ndim> tls;
    if (argview.shape.sample_dim() == 0) {
      if (ndim < 0)
        ndim = 1;
      out_tls.resize(N, ndim);
      for (int i = 0; i < N; i++)
        for (int d = 0; d < ndim; d++)
          tls.tensor_shape_span(i)[d] = to_extent(*argview.data[i]);
    } else {
      DALI_ENFORCE(argview.shape.sample_dim() == 1, "Shapes must be 1D tensors with extent equal "
       "to shape dimensionality (or scalar)");
      int D = argview.shape[0][0];
      DALI_ENFORCE(ndim < 0 || D == ndim, make_string(D, "-element tensor cannot describe an ",
        ndim, "D shape."));
      out_tls.resize(N, D);
      for (int i = 0; i < N; i++)
        for (int d = 0; d < D; d++)
          out_tls.tensor_shape_span(i)[d] = to_extent(argview.data[i][d]);
    }
  } else {
    // If the argument is specified as a vector, it represents a uniform tensor list shape.
    std::vector<ArgumentType> tsvec;
    if (ndim > 0) {
      // we have the luxury of knowing ndim ahead of time, so we can broadcast a scalar
      GetSingleOrRepeatedArg<ArgumentType>(spec, tsvec, argument_name, ndim);
    } else {
      // in dynamic use case, we get the dimensionality from the number of values
      tsvec = spec.GetRepeatedArgument<ArgumentType>(argument_name);
    }
    TensorShape<out_ndim> sample_shape;
    sample_shape.resize(tsvec.size());
    for (int i = 0; i < sample_shape.size(); i++)
      sample_shape[i] = to_extent(tsvec[i]);
    out_tls = uniform_list_shape<out_ndim>(batch_size, sample_shape);
  }
}

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_COMMON_H_
