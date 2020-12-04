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

template <typename Backend>
struct RNGBaseFields;

template <typename Backend, typename Impl>
class RNGBase : public Operator<Backend> {
 public:
  ~RNGBase() override = default;

 protected:
  explicit RNGBase(const OpSpec &spec)
      : Operator<Backend>(spec),
        rng_(spec.GetArgument<int64_t>("seed"), batch_size_),
        backend_specific_(spec.GetArgument<int64_t>("seed"), batch_size_) {
  }

  Impl &This() noexcept { return static_cast<Impl&>(*this); }
  const Impl &This() const noexcept { return static_cast<const Impl&>(*this); }

  bool CanInferOutputs() const override {
    return true;
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc,
                 const workspace_t<Backend> &ws) override {
    if (!spec_.TryGetArgument(dtype_, "dtype"))
      dtype_ = This().DefaultDataType();

    bool has_shape = spec_.ArgumentDefined("shape");
    bool has_shape_like = spec_.NumRegularInput() == 1;
    DALI_ENFORCE(!(has_shape && has_shape_like),
      "Providing argument \"shape\" is incompatible with providing a shape-like input");

    if (has_shape_like) {
      if (ws.template InputIsType<Backend>(0)) {
        shape_ = ws.template InputRef<Backend>(0).shape();
      } else if (std::is_same<GPUBackend, Backend>::value && ws.template InputIsType<CPUBackend>(0)) {
        shape_ = ws.template InputRef<CPUBackend>(0).shape();
      } else {
        DALI_FAIL("Shape-like input can be either CPUBackend or GPUBackend for case of GPU operators.");
      }
    } else if (has_shape) {
      GetShapeArgument(shape_, spec_, "shape", ws);
    } else {
      shape_ = uniform_list_shape(spec_.template GetArgument<DALIDataType>("batch_size"), {1});
    }
    batch_size_ = shape_.size();
    single_value_ = shape_.num_elements() == batch_size_;

    This().AcquireArgs(spec_, ws, batch_size_);

    output_desc.resize(1);
    output_desc[0].shape = shape_;
    output_desc[0].type = TypeTable::GetTypeInfo(dtype_);
    return true;
  }

  template <typename T>
  void RunImplTyped(workspace_t<GPUBackend> &ws);

  template <typename T>
  void RunImplTyped(workspace_t<CPUBackend> &ws);

  using Operator<Backend>::spec_;
  using Operator<Backend>::batch_size_;

  DALIDataType dtype_;
  BatchRNG<std::mt19937_64> rng_;
  TensorListShape<> shape_;
  bool single_value_ = false;

  RNGBaseFields<Backend> backend_specific_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_RNG_BASE_H_
