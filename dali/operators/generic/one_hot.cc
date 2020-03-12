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

#include "dali/operators/generic/one_hot.h"

namespace dali {

#define PREEMPH_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float, double)  // NOLINT

void OneHot::RunImpl(HostWorkspace &ws) {
  const auto &input = ws.template InputRef<CPUBackend>(0);
  auto &output = ws.template OutputRef<CPUBackend>(0);
  auto &tp = ws.GetThreadPool();
  TYPE_SWITCH(input.type().id(), type2id, InputType, PREEMPH_TYPES, (
          TYPE_SWITCH(output_type_, type2id, OutputType, PREEMPH_TYPES, (
          for (int sample_id = 0; sample_id < batch_size_; ++sample_id) {
            tp.DoWorkWithID(
                    [&, sample_id](int thread_id) {
                        OneHot(out, in, sample_id);
                    });
          }
  ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())))  // NOLINT
  tp.WaitForWork();
}

template <typename Out, typename In>
void OneHot(const OutListCPU<Out> &out, const InListCPU<In> &in, int sample_id) {
  // TODO
  auto &inptr = in[sample_id];
  // CHECK
  auto cls = inptr.template mutable_data<int>();
  auto &outptr = out[sample_id];
  // CHECK
  auto one_hot = outptr.template mutable_data<Out>();
  if (cls < nclasses_ && cls >= 0) {
    one_hot[*cls] = 1;
  }
}

DALI_REGISTER_OPERATOR(OneHot, OneHot, CPU);

DALI_SCHEMA(OneHot)
    .DocStr(
        "Produce tensor representing one hot encoding "
        " of the given input")
    .NumInput(1)
    .NumOutput(1)
    .AddOptionalArg("nclasses", R"code(Number of all classes)code", 0);
    .AddOptionalArg(arg_names::kDtype, R"code(Data type for the output)code",
                    DALI_FLOAT);

}  // namespace dali