// Copyright (c) 2017-2020, NVIDIA CORPORATION. All rights reserved.
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

#include <string>
#include <utility>
#include <vector>

#include "dali/core/error_handling.h"
#include "dali/core/tensor_shape.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/operator/op_spec.h"
#include "dali/pipeline/workspace/workspace.h"

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
void GetPerSampleArgument(std::vector<T> &output, const std::string &argument_name,
                          const OpSpec &spec, const ArgumentWorkspace &ws, int batch_size) {
  DALI_ENFORCE(batch_size >= 0,
               make_string("Invalid batch size. Expected nonnegative, actual: ", batch_size));
  if (spec.HasTensorArgument(argument_name)) {
    const auto &arg = ws.ArgumentInput(argument_name);
    decltype(auto) shape = arg.shape();
    int N = shape.num_samples();
    if (N == 1) {
      bool is_valid_shape = volume(shape.tensor_shape(0)) == batch_size;

      DALI_ENFORCE(is_valid_shape, make_string("`", argument_name, "` must be a 1xN or Nx1 (N = ",
                                               batch_size, ") tensor list. Got: ", shape));

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

/**
 * @brief Fill the result span with the argument which can be provided as:
 * * ArgumentInput - {result.size()}-shaped Tensor
 * * ArgumentInput - {} or {1}-shaped Tensor, the value will be replicated `result.size()` times
 * * Vector input - single "repeated argument" of length `result.size()` or 1
 * * scalar argument - it will be replicated `result.size()` times
 *
 * TODO(klecki): we may want to make this a generic utility and propagate the span-approach to
 * the rest of the related argument gettters
 *
 * TODO(michalz): rework it somehow to avoid going through this logic for each sample
 */
template <typename T>
void GetGeneralizedArg(span<T> result, const std::string &name, int sample_idx, const OpSpec& spec,
                       const ArgumentWorkspace& ws) {
  int argument_length = result.size();
  if (spec.HasTensorArgument(name)) {
    const auto& tv = ws.ArgumentInput(name);
    const auto& tensor = tv[sample_idx];
    const auto& shape = tensor.shape();
    auto vol = volume(shape);
    if (shape.size() != 0) {
      DALI_ENFORCE(shape.size() == 1,
                   make_string("Argument ", name, " for sample ", sample_idx,
                               " is expected to be a scalar or a 1D tensor, got: ",
                               shape.size(), "D."));
      DALI_ENFORCE(vol == 1 || vol == argument_length,
                   make_string("Argument ", name, " for sample ", sample_idx,
                               " is expected to have 1 or ", argument_length,
                               "elements, got: ", shape[0], "."));
    }
    if (vol == 1) {
      for (int i = 0; i < argument_length; i++) {
        result[i] = tensor.data<T>()[0];
      }
    } else {
      memcpy(result.data(), tensor.data<T>(), sizeof(T) * argument_length);
    }
    return;
  }
  std::vector<T> tmp;
  // we already handled the argument input, this handles spec-related arguments only
  GetSingleOrRepeatedArg(spec, tmp, name, argument_length);
  memcpy(result.data(), tmp.data(), sizeof(T) * argument_length);
}

template <typename ArgumentType, typename ExtentType>
std::pair<int, int> GetShapeLikeArgument(std::vector<ExtentType> &out_shape, const OpSpec &spec,
                                         const std::string &argument_name,
                                         const ArgumentWorkspace &ws, int batch_size,
                                         int ndim = -1) {
  DALI_ENFORCE(batch_size >= 0,
               make_string("Invalid batch size. Expected nonnegative, actual: ", batch_size));

  auto to_extent = [](ArgumentType extent) {
    return std::is_floating_point<ArgumentType>::value && std::is_integral<ExtentType>::value
               ? static_cast<ExtentType>(std::lround(extent))
               : static_cast<ExtentType>(extent);
  };

  if (spec.HasTensorArgument(argument_name)) {
    const auto &arg = ws.ArgumentInput(argument_name);
    auto argview = view<const ArgumentType>(arg);
    int N = argview.shape.num_samples();
    DALI_ENFORCE(N == batch_size, make_string("Unexpected number of samples in argument `",
      argument_name, "` (expected: ", batch_size, ")"));
    DALI_ENFORCE(is_uniform(argview.shape), "A tensor list shape must have the same dimensionality "
      "for all samples.");
    if (argview.shape.sample_dim() == 0) {  // a list of true scalars (0D)
      if (ndim < 0)  // no ndim? assume 1D
        ndim = 1;
      out_shape.resize(N * ndim);
      for (int i = 0; i < N; i++) {
        auto e = to_extent(*argview.data[i]);
        for (int d = 0; d < ndim; d++)
          out_shape[i * ndim + d] = e;  // broadcast scalar size to all dims
      }
    } else {
      DALI_ENFORCE(argview.shape.sample_dim() == 1, "Shapes must be 1D tensors with extent equal "
       "to shape dimensionality (or scalar)");
      int D = argview.shape[0][0];
      DALI_ENFORCE(ndim < 0 || D == ndim, make_string(D, "-element tensor cannot describe an ",
        ndim, "D shape."));
      ndim = D;
      out_shape.resize(N * D);
      for (int i = 0; i < N; i++)
        for (int d = 0; d < D; d++)
          out_shape[i * D + d] = to_extent(argview.data[i][d]);
    }
  } else {
    // If the argument is specified as a vector, it represents a uniform tensor list shape.
    std::vector<ArgumentType> tsvec;
    if (ndim >= 0) {
      // we have the luxury of knowing ndim ahead of time, so we can broadcast a scalar
      GetSingleOrRepeatedArg<ArgumentType>(spec, tsvec, argument_name, ndim);
    } else {
      // in dynamic use case, we get the dimensionality from the number of values
      tsvec = spec.GetRepeatedArgument<ArgumentType>(argument_name);
      ndim = tsvec.size();
    }
    out_shape.resize(batch_size * ndim);
    for (int i = 0; i < batch_size; i++)
      for (int d = 0; d < ndim; d++)
        out_shape[i * ndim + d] = to_extent(tsvec[d]);
  }

  return {batch_size, ndim};
}

template <typename ArgumentType = int, int out_ndim>
std::pair<int, int> GetShapeArgument(TensorListShape<out_ndim> &out_tls, const OpSpec &spec,
                                     const std::string &argument_name, const ArgumentWorkspace &ws,
                                     int batch_size, int ndim = out_ndim) {
  auto ret =
      GetShapeLikeArgument<ArgumentType>(out_tls.shapes, spec, argument_name, ws, batch_size, ndim);
  out_tls.resize(ret.first, ret.second);
  return ret;
}

}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_COMMON_H_
