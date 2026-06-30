// Copyright (c) 2017-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_PIPELINE_OPERATOR_BUILTIN_MAKE_CONTIGUOUS_H_
#define DALI_PIPELINE_OPERATOR_BUILTIN_MAKE_CONTIGUOUS_H_

#include <algorithm>
#include <vector>
#include <utility>

#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/pipeline/operator/common.h"
#include "dali/core/common.h"

// Found by benchmarking coalesced vs non coalesced on diff size images
#define COALESCE_THRESHOLD 8192

namespace dali {

enum class MakeContiguousMode {
  AlwaysCopy,       //! Always perform a copy.
  PassThrough,      //! Never copy.
  Opportunistic,    //! If already contiguous, pass through; otherwise copy.
  PipelineOutput,   //! Opportunistic + check against iteration's UnshareableData
};

template<typename Backend>
class MakeContiguousBase : public StatelessOperator<Backend> {
 public:
  inline explicit MakeContiguousBase(const OpSpec &spec) :
      StatelessOperator<Backend>(spec) {
    std::vector<int> hints;
    GetSingleOrRepeatedArg(spec, hints, "bytes_per_sample_hint", spec.NumOutput());
    if (!hints.empty())
      bytes_per_sample_hint = hints[0];
  }

  virtual inline ~MakeContiguousBase() = default;

  template <typename InputBackend>
  void SetPassthrough(const Workspace &ws, const TensorList<InputBackend> &input) {
    bool is_contiguous = input.IsContiguousInMemory();
    pass_through_ =
      mode_ == MakeContiguousMode::PassThrough ||
      ((mode_ == MakeContiguousMode::Opportunistic || mode_ == MakeContiguousMode::PipelineOutput)
        && is_contiguous);
    if (pass_through_ && mode_ == MakeContiguousMode::PipelineOutput) {
      auto &unshareable = ws.GetIterationData()->unshareable_data;
      auto lock = unshareable.Lock();
      if (!unshareable.Empty()) {
        if (is_contiguous) {
          if (unshareable.Contains(input.raw_tensor(0)))
            pass_through_ = false;
        } else {
          for (int i = 0; i < input.num_samples(); i++) {
            if (unshareable.Contains(input.raw_tensor(i))) {
              pass_through_ = false;
              break;
            }
          }
        }
      }
    }
  }

  bool SetupImpl(std::vector<OutputDesc> &output_desc, const Workspace &ws) override {
    output_desc.resize(1);
    bool is_contiguous = false;

    if (ws.InputIsType<CPUBackend>(0)) {
      auto &input = ws.Input<CPUBackend>(0);
      output_desc[0].shape = input.shape();
      output_desc[0].type = input.type();
      SetPassthrough(ws, input);
    } else {
      auto &input = ws.Input<GPUBackend>(0);
      output_desc[0].shape = input.shape();
      output_desc[0].type = input.type();
      SetPassthrough(ws, input);
    }

    return !pass_through_;
  }

  DISABLE_COPY_MOVE_ASSIGN(MakeContiguousBase);

  /**
   * @brief Intended to be called by the executor. If the executor guarantees that the input
   * to the make contiguous is always contiguous and we can safely pass through the data.
   * The executor is responsible for adjusting the prefetch queue sizes of the passed through
   * inputs.
   */
  void MarkPassThrough() {
    mode_ = MakeContiguousMode::PassThrough;
  }

  /**
   * @brief Check if this MakeContiguous node is set to pass through data (or copy it).
   *
   * Result is valid after the executor runs the OpGraph::SetupMakeContiguousPassThrough pass
   * on the graph.
   */
  bool IsPassThrough() const {
    return mode_ == MakeContiguousMode::PassThrough;
  }

  void SetMode(MakeContiguousMode mode) {
    mode_ = mode;
  }

  MakeContiguousMode GetMode() const {
    return mode_;
  }

 protected:
  USE_OPERATOR_MEMBERS();
  TensorList<CPUBackend> cpu_output_buff;
  bool coalesced = true;
  // Whether the next batch would be passed through - this value is changed in Setup.
  bool pass_through_ = false;
  int bytes_per_sample_hint = 0;
  MakeContiguousMode mode_ = MakeContiguousMode::AlwaysCopy;
};


class MakeContiguousGPU : public MakeContiguousBase<GPUBackend> {
 public:
  inline explicit MakeContiguousGPU(const OpSpec &spec) :
      MakeContiguousBase<GPUBackend>(spec) {}

  using Operator<GPUBackend>::RunImpl;
  void RunImpl(Workspace &ws) override;
  DISABLE_COPY_MOVE_ASSIGN(MakeContiguousGPU);
};

class MakeContiguousMixed : public MakeContiguousBase<MixedBackend> {
 public:
  inline explicit MakeContiguousMixed(const OpSpec &spec) :
      MakeContiguousBase<MixedBackend>(spec) {}
  using Operator<MixedBackend>::Run;

  void RunImpl(Workspace &ws) override;

  DISABLE_COPY_MOVE_ASSIGN(MakeContiguousMixed);
};

class MakeContiguousCPU : public MakeContiguousBase<CPUBackend> {
 public:
  inline explicit MakeContiguousCPU(const OpSpec &spec) :
      MakeContiguousBase<CPUBackend>(spec) {}

  using Operator<CPUBackend>::RunImpl;
  void RunImpl(Workspace &ws) override;
  DISABLE_COPY_MOVE_ASSIGN(MakeContiguousCPU);
};

/**
 * @brief Call the MakeContiguousBase::MarkPassThrough, invalid for other operators.
 */
void MarkPassThrough(OperatorBase &make_contiguous);

/**
 * @brief Call the MakeContiguousBase::IsPassThrough, invalid for other operators.
 */
bool IsPassThrough(const OperatorBase &make_contiguous);

/**
 * @brief Call the MakeContiguousBase::SetMode, invalid for other operators.
 *
 * @return true, if the operator was MakeContiguous and the mode was set, false otherwise.
 */
bool SetMakeContiguousMode(OperatorBase &make_contiguous, MakeContiguousMode mode);


}  // namespace dali

#endif  // DALI_PIPELINE_OPERATOR_BUILTIN_MAKE_CONTIGUOUS_H_
