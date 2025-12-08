// Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cassert>
#include <random>
#include <string>
#include <vector>
#include <memory>
#include <cstdio>
#include <stdexcept>

#include "dali/core/convert.h"
#include "dali/pipeline/data/types.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/operator/checkpointing/op_checkpoint.h"
#include "dali/core/static_switch.h"
#include "dali/operators/random/philox.h"

namespace dali {
namespace rng {

template <typename Backend>
struct OperatorWithRngFields;

/** The step taken between adjacent elements within a sample.
 *
 * It may be necessary to generate several uniform uint32 numbers to produce a single element of
 * the output. This number is chosen to be large enough to avoid correlation between the elements,
 * while being mutually prime with 2^64 so it will not reduce the space of possible states.
 * It is a Fermat number, so multiplication is likely to be optimized into a shift and addition.
 */
static constexpr int kSkipaheadPerElement = 257;
/** If we want to advance to a next subsequence within a sample.
 *
 * It may be desierable to generate several subsequences of random numbers to produce a single
 * sample of the output. This number is chosen to be large enough to avoid correlation between the
 * subsequences, while being mutually prime with 2^64 so it will not reduce the space of possible
 * states.
 * It is a Fermat number, so multiplication is likely to be optimized into a shift and addition.
 */
static constexpr int kSkipaheadPerSample = 65537;

void _DetectOperatorBackend(int /* ... */);
template <typename Backend>
Backend _DetectOperatorBackend(const Operator<Backend> *);

template <typename Operator>
struct backend_type {
  using type = decltype(_DetectOperatorBackend(std::declval<Operator *>()));
};

template <typename X>
using backend_t = typename backend_type<X>::type;

template <typename Base>
class OperatorWithRng : public Base {
 public:
  using Backend = backend_t<Base>;

  void SaveState(OpCheckpoint &cpt, AccessOrder order) override {
    cpt.MutableCheckpointState() = master_rng_.get_state();
  }

  void RestoreState(const OpCheckpoint &cpt) override {
    master_rng_.set_state(cpt.CheckpointState<Philox4x32_10::State>());
  }

  std::string SerializeCheckpoint(const OpCheckpoint &cpt) const override {
    const auto &state = cpt.CheckpointState<Philox4x32_10::State>();
    return Philox4x32_10::state_to_string(state);
  }

  void DeserializeCheckpoint(OpCheckpoint &cpt, const std::string &data) const override {
    Philox4x32_10::State s;
    Philox4x32_10::state_from_string(s, data);
    cpt.MutableCheckpointState() = s;
  }

  void Run(Workspace &ws) override {
    if (has_random_state_arg_) {
      LoadRandomState(ws);
    }
    Base::Run(ws);
    assert(ws.NumOutput() > 0);
    Advance(ws.GetOutputBatchSize(0));
  }

  Philox4x32_10 GetSampleRNG(int sample_idx) const {
    Philox4x32_10 rng = master_rng_;
    rng.skipahead_sequence(sample_idx * kSkipaheadPerSample);
    return rng;
  }

 protected:
  explicit OperatorWithRng(const OpSpec &spec)
      : Base(spec)
      , has_random_state_arg_(spec.HasTensorArgument("_random_state")) {
    int64_t seed = spec.GetArgument<int64_t>("seed");
    master_rng_.init(seed, 0, 0);
  }

  void LoadRandomState(const Workspace &ws) {
    const TensorList<CPUBackend> &random_state = ws.ArgumentInput("_random_state");
    assert(random_state.num_samples() > 0);
    int element_size = random_state.type_info().size();
    if (random_state[0].shape().num_elements() * element_size < 25)
      throw std::invalid_argument("Random state tensor is too small");
    const char *state_data = static_cast<const char *>(random_state[0].raw_data());
    Philox4x32_10::State state;
    memcpy(&state.key, state_data, 8);
    memcpy(&state.ctr, state_data + 8, 16);
    memcpy(&state.phase, state_data + 24, 1);
    state.phase &= 3;
    master_rng_.set_state(state);
  }

  inline void Advance(int batch_size) {
    master_rng_.skipahead_sequence(batch_size);
  }

  using Base::max_batch_size_;
  using Base::spec_;

  Philox4x32_10 master_rng_;
  bool has_random_state_arg_ = false;
};

/**
 * @brief CRTP class for implementing random number and noise generators.
 *
 * @tparam IsNoiseGen - noise generators by default copy type from input and compute the output
 * value based on the input value at given coordinate.
 */
template <typename Backend, typename Impl, bool IsNoiseGen>
class RNGBase : public OperatorWithRng<Operator<Backend>> {
 protected:
  using Base = OperatorWithRng<Operator<Backend>>;
  explicit RNGBase(const OpSpec &spec)
      : Base(spec)
      , backend_data_(NumDists()) {}

  Impl &This() noexcept { return static_cast<Impl&>(*this); }
  const Impl &This() const noexcept { return static_cast<const Impl&>(*this); }

  /** @defgroup RngCRTP Customization points for random number generators.
   *  @{
   */

  /**
   * @brief Return the default data type of output if the "dtype" argument is not specified.
   */
  DALIDataType DefaultDataType(const OpSpec &spec, const Workspace &ws) const {
    return DALI_NO_TYPE;
  }

  /**
   * @brief Customization point for obtaining argument inputs or arguments during SetupImpl.
   */
  void AcquireArgs(const OpSpec &spec, const Workspace &ws, int nsamples) {}

  /**
   * @brief If the output shape is non copied from `shape` argument or `__shape_like` input,
   * return new one that should be used instead. Called after basic shape parameter is obtained
   * into `shape_`.
   */
  TensorListShape<> PostprocessShape(const OpSpec &spec, const Workspace &ws) {
    return shape_;
  }

  /**
   * @brief The RNG implementation must provide SetupDist function, with `Dist` type
   * and call RunImplTyped<T, Dist>(ws) in its RunImpl, for given output type `T`.
   *
   * The type `Dist` must implement function used for sampling based on random generator for
   * given device with some output type `U`.
   * The signature for random generators:
   *   template <typename Generator>
   *   DALI_HOST_DEV U Generate(Generator &st)
   *
   * The signature for noise generators:
   *   template <typename Generator>
   *   DALI_HOST_DEV U Generate(T input, Generator &st)

   * @param dists_data
   * @param nsamples
   * @return true - if the Dist constructed by this function should be used
   * @return false - if the default-constructed one can be used - allowing to skip the copy to GPU.
   */
  template <typename Dist>
  bool SetupDists(Dist* dists_data, int nsamples);

  /** @} */  // end of RngCRTP

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
      dtype_ = This().DefaultDataType(spec_, ws);

    bool has_shape = spec_.ArgumentDefined("shape");
    // The first optional input is the shape like
    bool has_shape_like = spec_.NumRegularInput() == spec_.GetSchema().MinNumInput() + 1;
    int nsamples = GetBatchSize(ws);
    DALI_ENFORCE(!(has_shape && has_shape_like),
      "Providing argument \"shape\" is incompatible with providing a shape-like input");

    if (IsNoiseGen) {
      shape_ = ws.Input<Backend>(0).shape();
    } else if (has_shape_like) {
      int shape_like_idx = spec_.GetSchema().MinNumInput();
      shape_ = ws.GetInputShape(shape_like_idx);
    } else if (has_shape) {
      GetShapeArgument(shape_, spec_, "shape", ws, nsamples);
    } else {
      shape_ = uniform_list_shape(nsamples, TensorShape<0>{});
    }
    This().AcquireArgs(spec_, ws, shape_.size());
    shape_ = This().PostprocessShape(spec_, ws);

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

  int NumDists() const {
    return max_batch_size_;
  }

  using Base::spec_;
  using Base::max_batch_size_;
  using Base::master_rng_;
  using Base::GetSampleRNG;

  DALIDataType dtype_ = DALI_NO_TYPE;
  TensorListShape<> shape_;
  OperatorWithRngFields<Backend> backend_data_;
};

}  // namespace rng
}  // namespace dali

#endif  // DALI_OPERATORS_RANDOM_RNG_BASE_H_
