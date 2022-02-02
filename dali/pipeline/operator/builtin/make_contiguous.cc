// Copyright (c) 2017-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/operator/builtin/make_contiguous.h"
#include "dali/core/access_order.h"
#include "dali/pipeline/data/backend.h"

namespace dali {

void MakeContiguousCPU::RunImpl(HostWorkspace &ws) {
  auto &input = ws.template Input<CPUBackend>(0);
  auto &output = ws.template Output<CPUBackend>(0);
  int batch_size = input.num_samples();
  output.SetLayout(input.GetLayout());
  const auto &shapes = input.shape();
  const auto &type = input.type_info();

  auto &thread_pool = ws.GetThreadPool();
  for (int sample_id = 0; sample_id < batch_size; ++sample_id) {
    thread_pool.AddWork([sample_id, &input, &output, &type, &shapes] (int tid) {
      // TODO, tbh, this is bonkers that we go through resize twice, once in setup, and once
      // with the "rich copy" here:
      // output[sample_id].Copy(input[sample_id], AccessOrder::host());  // todo view<void>
      output.CopySample(sample_id, input, sample_id, AccessOrder::host());

      // if (!order)
      //   order = other.order() ? other.order() : order_;
      // this->Resize(other.shape(), other.type());
      // order.wait(order_);
      // this->SetLayout(other.GetLayout());
      // this->SetSourceInfo(other.GetSourceInfo());
      // this->SetSkipSample(other.ShouldSkipSample());
      // type_.template Copy<Backend, InBackend>(this->raw_mutable_data(),
      //     other.raw_data(), this->size(), order.stream());
      // order_.wait(order);

      // todo: this is probably not necessary
      // auto order = AccessOrder::host();
      // order.wait(output.order());
      // type.template Copy<CPUBackend, CPUBackend>(
      //     output.raw_mutable_tensor(sample_id), input.raw_tensor(sample_id),
      //     shapes[sample_id].num_elements(), order.stream());
      // output.order().wait(order);
      // auto out_meta = output.GetMeta(sample_id);
      // const auto &in_meta = input.GetMeta(sample_id);
      // out_meta.SetSourceInfo(in_meta.GetSourceInfo());
      // out_meta.SetSkipSample(in_meta.ShouldSkipSample());
      // output.SetMeta(sample_id, out_meta);
    }, shapes.tensor_size(sample_id));
  }
  thread_pool.RunAll();
}

DALI_REGISTER_OPERATOR(MakeContiguous, MakeContiguousCPU, CPU);

DALI_SCHEMA(MakeContiguous)
  .DocStr(R"code(Move input batch to a contiguous representation, more suitable for execution on the GPU)code")
  .NumInput(1)
  .NumOutput(1)
  .MakeInternal();

}  // namespace dali
