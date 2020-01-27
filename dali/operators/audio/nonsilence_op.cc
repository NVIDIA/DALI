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

#include <dali/kernels/signal/dct/dct_cpu.h>
#include <dali/pipeline/data/views.h>
#include "dali/operators/audio/nonsilence_op.h"
#include "dali/core/static_switch.h"
#include "dali/core/convert.h"

namespace dali {

DALI_SCHEMA(NonsilenceRegion)
                .DocStr(R"code(The operator performs leading and trailing silence detection in an audio buffer.<br>
This operators' behaviour can be described as:
```
def nonsilence(buffer, cutoff_value):
    begin = 0
    end = 0
    for i in range(len(buffer)):
        if buffer[i] > cutoff_value:
            begin = i
            break
    for i in range(len(buffer) - 1, -1, -1):
        if buffer[i] > cutoff_value:
            end = i
            break
    length = end - begin + 1
    return begin, length
```
`Input`: 1-D audio buffer
`Output[0]`: Begin index of nonsilent region
`Output[1] >= 0`: Length of nonsilent region<br>
If `Output[1] == 0`, `Output[0]` value is undefined
)code")
                .NumInput(1)
                .NumOutput(detail::kNumOutputs)
                .AddArg(detail::kCutoff,
                        R"code(Everything below this value will be regarded as silence)code",
                        DALI_FLOAT);

DALI_REGISTER_OPERATOR(NonsilenceRegion, NonsilenceOperatorCpu, CPU);



//bool NonsilenceOperatorCpu::SetupImpl(std::vector<OutputDesc> &output_desc,
//                                      const workspace_t<CPUBackend> &ws) {
//  TypeInfo output_type;
//  output_type.SetType<detail::OutputType>(TypeTable::GetTypeID<detail::OutputType>());
//  TensorShape<> scalar_shape = {1};
//
//  output_desc.resize(detail::kNumOutputs);
//  for (int i = 0; i < detail::kNumOutputs; i++) {
//    output_desc[i].shape = uniform_list_shape(batch_size_, scalar_shape);
//    output_desc[i].type = output_type;
//  }
//  return true;
//}


//template<typename InputType>
//void NonsilenceOperatorCpu::RunImplTyped(workspace_t<CPUBackend> &ws) {
//  const auto &input = ws.template InputRef<CPUBackend>(0);
//  auto &output_begin = ws.OutputRef<CPUBackend>(0);
//  auto &output_length = ws.OutputRef<CPUBackend>(1);
//  auto &tp = ws.GetThreadPool();
//  int nsamples = input.size();
//  auto nthreads = ws.GetThreadPool().size();




//  int sample_id=0;


//  const auto in_view = view<const InputType, 1>(input[sample_id]);

//  const auto in_ptr = input[sample_id].data<InputType>();
//  auto num_samples = volume(input[sample_id].shape());
//  auto res = detail::DetectNonsilenceRegion
//          (make_cspan(in_ptr, num_samples), ConvertSat<InputType>(cutoff_));
//  auto beg_ptr = output_begin[sample_id].mutable_data<detail::OutputType>();
//  auto len_ptr = output_length[sample_id].mutable_data<detail::OutputType>();
//  *beg_ptr = res.first;
//  *len_ptr = res.second;

//  for (int sample_id = 0; sample_id < batch_size_; sample_id++) {
//    tp.DoWorkWithID(
//            [&, sample_id](int thread_id) {
//                const auto in_ptr = input[sample_id].data<InputType>();
//                auto num_samples = volume(input[sample_id].shape());
//                auto res = detail::DetectNonsilenceRegion
//                        (make_cspan(in_ptr, num_samples), ConvertSat<InputType>(cutoff_));
//                auto beg_ptr = output_begin[sample_id].mutable_data<detail::OutputType>();
//                auto len_ptr = output_length[sample_id].mutable_data<detail::OutputType>();
//                *beg_ptr = res.first;
//                *len_ptr = res.second;
//            });
//  }

//  tp.WaitForWork();
//}

#define NONSILENCE_TYPES (uint8_t, int8_t, uint16_t, int16_t, uint32_t, int32_t, uint64_t, int64_t, float, double)  // NOLINT

//void NonsilenceOperatorCpu::RunImpl(workspace_t<CPUBackend> &ws) {
//  const auto &input = ws.template InputRef<CPUBackend>(0);
//  TYPE_SWITCH(input.type().id(), type2id, InputType, NONSILENCE_TYPES, (
//          RunImplTyped<InputType>(ws);
//  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())))  // NOLINT
//}

#undef NONSILENCE_TYPES

//template<typename T>
//std::pair<int, int> NonsilenceOperatorCpu::DetectNonsilenceRegion(TensorView<CPUBackend, const T, 1>, T cutoff) {
//}

//template<typename Kernel>
//void NonsilenceOperatorCpuImpl::SetupKernel(int nthreads, int nsamples) {
//  kernel_manager_.Initialize<Kernel>();
//  kernel_manager_.Resize(nthreads, nsamples);
//}

//template<typename InputType, typename Kernel>
//void NonsilenceOperatorCpuImpl::RunKernel(TensorView<CPUBackend, const InputType, 1> in, int nsamples, int nthreads, int sample_id, int thread_id) {
//
//  kernels::KernelContext kctx;
//
//  auto reqs = kernel_manager_.Setup<Kernel>(sample_id, kctx, in);
//  intermediate_buffers_[sample_id].Resize(reqs.output_shapes[0][sample_id]);
//  auto out = view_as_tensor<float>(intermediate_buffers_[sample_id]);
//  kernel_manager_.Run(thread_id, sample_id, kctx, out, in, {2048,-1});
//  cout<<"JUZ\n";
//
//}












//template<typename InputType>
//std::pair<int, int> NonsilenceOperatorCpuImpl::DetectNonsilenceRegion(int thread_id, int sample_id,
//                                           TensorView<StorageCPU, const InputType, 1> in,
//                                           InputType cutoff) {
//  SetupKernels(nthreads, nsamples);
//  RunKernels(thread_id, sample_id, in, {3, -1}, {}); //TODO ARGS
//  auto dbs = view_as_tensor<float>(to_db_kernel_.outputs_[sample_id]);
//  return LeadTrailThresh(make_cspan(dbs.data, dbs.num_elements()), cutoff);
//
//}





//template<typename T>
//std::pair<int, int> NonsilenceOperatorCpuImpl::LeadTrailThresh(span<const T> buffer, T cutoff) {
//  assert(buffer.size()>0);
//  int begin = -1;
//  int end = buffer.size();
//  while (begin < end && buffer[++begin] < cutoff);  // NOLINT
//  if (begin == end) return {-1, 0};
//  while (buffer[--end] < cutoff);  // NOLINT
//  return {begin, end - begin + 1};
//}



//void NonsilenceOperatorCpuImpl::SetupKernels(int nthreads, int nsamples) {
//  mms_kernel_.Setup(nthreads, nsamples);
//  to_db_kernel_.Setup(nthreads, nsamples);
//}


//template<typename InputType>
//void NonsilenceOperatorCpuImpl::RunKernels(int thread_id, int sample_id, TensorView<StorageCPU, const InputType, 1> in,
//                const MmsArgs &mms_args, const DbArgs &db_args) {
//  mms_kernel_.Run(thread_id, sample_id, in, mms_args);
//  auto db_in = view_as_tensor<const float>(mms_kernel_.outputs_[sample_id]).to_static<1>();
//  to_db_kernel_.Run(thread_id, sample_id, db_in, db_args);
//}









}  // namespace dali
