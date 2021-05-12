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

#ifndef DALI_OPERATORS_RANDOM_RNG_BASE_H_
#define DALI_OPERATORS_RANDOM_RNG_BASE_H_

#include <random>
#include <vector>
#include "dali/core/convert.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/util/batch_rng.h"
#include "dali/core/static_switch.h"
#include "dali/operators/util/randomizer.cuh"

namespace dali {

template <typename Backend, bool IsNoiseGen>
struct RNGBaseFields;

template <typename Backend, typename Impl, bool IsNoiseGen>
class RNGBase : public Operator<Backend> {
 protected:
  explicit RNGBase(const OpSpec &spec)
      : Operator<Backend>(spec),
        rng_(spec.GetArgument<int64_t>("seed"), max_batch_size_),
        backend_data_(spec.GetArgument<int64_t>("seed"), max_batch_size_) {
  }

  Impl &This() noexcept { return static_cast<Impl&>(*this); }
  const Impl &This() const noexcept { return static_cast<const Impl&>(*this); }

  bool CanInferOutputs() const override {
    return true;
  }

  int GetBatchSize(const workspace_t<Backend> &ws) const {
    if (spec_.NumRegularInput() == 1)
      return ws.template InputRef<Backend>(0).shape().size();
    else
      return ws.GetRequestedBatchSize(0);
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc,
                 const workspace_t<Backend> &ws) override {
    if (IsNoiseGen)
      dtype_ = ws.template InputRef<Backend>(0).type().id();
    else if (!spec_.TryGetArgument(dtype_, "dtype"))
      dtype_ = This().DefaultDataType();

    bool has_shape = spec_.ArgumentDefined("shape");
    bool has_shape_like = spec_.NumRegularInput() == 1;
    int nsamples = GetBatchSize(ws);
    DALI_ENFORCE(!(has_shape && has_shape_like),
      "Providing argument \"shape\" is incompatible with providing a shape-like input");

    if (IsNoiseGen) {
      shape_ = ws.template InputRef<Backend>(0).shape();
    } else if (has_shape_like) {
      if (ws.template InputIsType<Backend>(0)) {
        shape_ = ws.template InputRef<Backend>(0).shape();
      } else if (std::is_same<GPUBackend, Backend>::value &&
                 ws.template InputIsType<CPUBackend>(0)) {
        shape_ = ws.template InputRef<CPUBackend>(0).shape();
      } else {
        DALI_FAIL(
            "Shape-like input can be either CPUBackend or GPUBackend for case of GPU operators.");
      }
    } else if (has_shape) {
      GetShapeArgument(shape_, spec_, "shape", ws, nsamples);
    } else {
      shape_ = uniform_list_shape(nsamples, TensorShape<0>{});
    }
    This().AcquireArgs(spec_, ws, shape_.size());

    output_desc.resize(1);
    output_desc[0].shape = shape_;
    output_desc[0].type = TypeTable::GetTypeInfo(dtype_);
    return true;
  }

  bool PerChannel() const {
    // By default generators don't interpret channel data, treating the data as a 1D array
    // If set to false by an implementation, the generation will occur once and will be applied
    // to all channels
    return true;
  }

  template <typename T, typename Dist>
  void RunImplTyped(workspace_t<CPUBackend> &ws);

  template <typename T, typename Dist>
  void RunImplTyped(workspace_t<GPUBackend> &ws);

  using Operator<Backend>::spec_;
  using Operator<Backend>::max_batch_size_;

  DALIDataType dtype_ = DALI_NO_TYPE;
  BatchRNG<std::mt19937_64> rng_;
  TensorListShape<> shape_;
  RNGBaseFields<Backend, IsNoiseGen> backend_data_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_RNG_BASE_H_
