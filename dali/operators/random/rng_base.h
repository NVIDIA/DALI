// Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <string>
#include <vector>
#include <memory>

#include "dali/core/convert.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/checkpointing/snapshot_serializer.h"
#include "dali/pipeline/util/batch_rng.h"
#include "dali/core/static_switch.h"
#include "dali/operators/util/randomizer.cuh"
#include "dali/operators/random/rng_checkpointing_utils.h"

namespace dali {
namespace rng {

template <typename Backend>
struct OperatorWithRngFields;


template<typename Backend, bool RngPerSample = true>
class OperatorWithRng : public Operator<Backend>{
 public:
  using CheckpointType = std::conditional_t<std::is_same_v<Backend, CPUBackend>,
                                            BatchRNG<std::mt19937_64>, curand_states>;
  using CheckpointUtils = RngCheckpointUtils<Backend, CheckpointType>;

  void SaveState(OpCheckpoint &cpt, AccessOrder order) override {
    if constexpr (std::is_same_v<Backend, CPUBackend>) {
      CheckpointUtils::SaveState(cpt, order, rng_);
    } else {
      static_assert(std::is_same_v<Backend, GPUBackend>);
      CheckpointUtils::SaveState(cpt, order, backend_data_.randomizer_);
    }
  }

  void RestoreState(const OpCheckpoint &cpt) override {
    if constexpr (std::is_same_v<Backend, CPUBackend>) {
      CheckpointUtils::RestoreState(cpt, rng_);
    } else {
      static_assert(std::is_same_v<Backend, GPUBackend>);
      CheckpointUtils::RestoreState(cpt, backend_data_.randomizer_);
    }
  }

  std::string SerializeCheckpoint(const OpCheckpoint &cpt) const override {
    return CheckpointUtils::SerializeCheckpoint(cpt);
  }

  void DeserializeCheckpoint(OpCheckpoint &cpt, const std::string &data) const override {
    CheckpointUtils::DeserializeCheckpoint(cpt, data);
  }

 protected:
  size_t RngsCount() {
    if constexpr (RngPerSample) {
      return max_batch_size_;
    } else {
      return 1;
    }
  }

  explicit OperatorWithRng(const OpSpec &spec)
      : Operator<Backend>(spec),
        rng_(spec.GetArgument<int64_t>("seed"), RngsCount()),
        backend_data_(spec.GetArgument<int64_t>("seed"), RngsCount()) {}

  using Operator<Backend>::max_batch_size_;
  using Operator<Backend>::spec_;

  BatchRNG<std::mt19937_64> rng_;
  OperatorWithRngFields<Backend> backend_data_;
};


template <typename Backend, typename Impl, bool IsNoiseGen>
class RNGBase : public OperatorWithRng<Backend> {
 protected:
  explicit RNGBase(const OpSpec &spec)
      : OperatorWithRng<Backend>(spec) {}

  Impl &This() noexcept { return static_cast<Impl&>(*this); }
  const Impl &This() const noexcept { return static_cast<const Impl&>(*this); }

  bool CanInferOutputs() const override {
    return true;
  }

  int GetBatchSize(const Workspace &ws) const {
    if (spec_.NumRegularInput() == 1)
      return ws.Input<Backend>(0).shape().size();
    else
      return ws.GetRequestedBatchSize(0);
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc,
                 const Workspace &ws) override {
    if (IsNoiseGen)
      dtype_ = ws.Input<Backend>(0).type();
    else if (!spec_.TryGetArgument(dtype_, "dtype"))
      dtype_ = This().DefaultDataType();

    bool has_shape = spec_.ArgumentDefined("shape");
    bool has_shape_like = spec_.NumRegularInput() == 1;
    int nsamples = GetBatchSize(ws);
    DALI_ENFORCE(!(has_shape && has_shape_like),
      "Providing argument \"shape\" is incompatible with providing a shape-like input");

    if (IsNoiseGen) {
      shape_ = ws.Input<Backend>(0).shape();
    } else if (has_shape_like) {
      if (ws.InputIsType<Backend>(0)) {
        shape_ = ws.Input<Backend>(0).shape();
      } else if (std::is_same<GPUBackend, Backend>::value &&
                 ws.InputIsType<CPUBackend>(0)) {
        shape_ = ws.Input<CPUBackend>(0).shape();
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
    output_desc[0].type = dtype_;
    return true;
  }

  bool PerChannel() const {
    // By default generators don't interpret channel data, treating the data as a 1D array
    // If set to false by an implementation, the generation will occur once and will be applied
    // to all channels
    return true;
  }

  template <typename T, typename Dist>
  void RunImplTyped(Workspace &ws, CPUBackend);

  template <typename T, typename Dist>
  void RunImplTyped(Workspace &ws, GPUBackend);

  template <typename T, typename Dist>
  void RunImplTyped(Workspace &ws) {
    RunImplTyped<T, Dist>(ws, Backend{});
  }

  using OperatorWithRng<Backend>::spec_;
  using OperatorWithRng<Backend>::max_batch_size_;
  using OperatorWithRng<Backend>::rng_;
  using OperatorWithRng<Backend>::backend_data_;

  DALIDataType dtype_ = DALI_NO_TYPE;
  TensorListShape<> shape_;
};

}  // namespace rng
}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_RNG_BASE_H_
