// Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/core/static_switch.h"
#include "dali/operators/audio/nonsilence_op.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(NonsilentRegion)
  .DocStr(R"code(Performs leading and trailing silence detection in an audio buffer.

The operator returns the beginning and length of the non-silent region by comparing the
short term power calculated for `window_length` of the signal with a silence cut-off threshold.
The signal is considered to be silent when the ``short_term_power_db`` is less than
the `cutoff_db`. where::

  short_term_power_db = 10 * log10( short_term_power / reference_power )

Unless specified otherwise, `reference_power` is the maximum power of the signal.

Inputs and outputs:

* **Input 0** - 1D audio buffer.
* **Output 0** - Index of the first sample in the nonsilent region.
* **Output 1** - Length of nonsilent region.

.. note::
  If ``Outputs[1] == 0``,  the value in ``Outputs[0]`` is undefined.

.. warning::
  At this moment, the 'gpu' backend of this operator is implemented in terms of the 'cpu'
  implementation. This results in a device-to-host copy of the inputs and a host-to-device copy of the
  outputs. While using the 'gpu' implementation of this operator doesn't add any performance
  benefit on its own, using it might make sense in order to enable moving preceding operations in the
  pipeline to the GPU.
)code")
  .NumInput(1)
  .NumOutput(detail::kNumOutputs)
  .AddOptionalArg("cutoff_db",
                  R"code(The threshold, in dB, below which the signal is considered silent.)code",
                  -60.f)
  .AddOptionalArg("window_length", R"code(Size of the sliding window used to calculate of
the short-term power of the signal.)code", 2048)
  .AddOptionalArg("reference_power",
                  R"code(The reference power that is used to convert the signal to dB.

If a value is not provided, the maximum power of the signal will be used as the reference.)code",
                  0.f)
  .AddOptionalArg("reset_interval",
                  R"code(The number of samples after which the moving mean average is recalculated
to avoid loss of precision.

If ``reset_interval == -1``, or the input type allows exact calculation, the average will not be
reset. The default value can be used for most of the use cases.)code",
                  8192);

class NonsilenceOperatorCpu : public NonsilenceOperator<CPUBackend> {
 public:
  explicit NonsilenceOperatorCpu(const OpSpec &spec) :
          NonsilenceOperator<CPUBackend>(spec) {
    intermediate_buffers_.resize(num_threads_);
    for (auto &b : intermediate_buffers_) {
      b.set_pinned(false);
    }
  }

  ~NonsilenceOperatorCpu() override = default;
  DISABLE_COPY_MOVE_ASSIGN(NonsilenceOperatorCpu);

 protected:
  void RunImpl(Workspace &ws) override {
    const auto &input = ws.Input<CPUBackend>(0);

    TYPE_SWITCH(input.type(), type2id, InputType, (NONSILENCE_TYPES),
      (RunImplTyped<InputType>(ws);),
      (DALI_FAIL(
          make_string("Unsupported input type: ", input.type(),
                      "\nSupported types are : ", ListTypeNames<NONSILENCE_TYPES>()));));
  }

 private:
  template<typename InputType>
  void RunImplTyped(Workspace &ws) {
    const auto &input = ws.Input<CPUBackend>(0);
    auto &output_begin = ws.Output<CPUBackend>(0);
    auto &output_length = ws.Output<CPUBackend>(1);
    assert(output_begin.sample_dim() == 0);
    assert(output_length.sample_dim() == 0);
    auto curr_batch_size = ws.GetInputBatchSize(0);
    auto &tp = ws.GetThreadPool();
    auto in_shape = input.shape();
    for (int sample_id = 0; sample_id < curr_batch_size; sample_id++) {
      tp.AddWork(
              [&, sample_id](int thread_id) {
                  detail::Args<InputType> args;
                  args.input = view<const InputType, 1>(input[sample_id]);
                  args.cutoff_db = cutoff_db_[sample_id];
                  if (!reference_max_) {
                    args.reference_power = reference_power_[sample_id];
                  }
                  args.reference_max = reference_max_;
                  args.window_length = std::min<int>(window_length_, args.input.num_elements());
                  args.reset_interval = reset_interval_;

                  auto res = DetectNonsilenceRegion(intermediate_buffers_[thread_id], args);
                  auto *beg_ptr = output_begin.mutable_tensor<int>(sample_id);
                  auto *len_ptr = output_length.mutable_tensor<int>(sample_id);
                  *beg_ptr = res.first;
                  *len_ptr = res.second;
              }, in_shape.tensor_size(sample_id));
    }
    tp.RunAll();
  }

  std::vector<Tensor<CPUBackend>> intermediate_buffers_;
};

DALI_REGISTER_OPERATOR(NonsilentRegion, NonsilenceOperatorCpu, CPU);


}  // namespace dali
