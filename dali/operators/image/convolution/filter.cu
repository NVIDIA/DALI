// Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <memory>

#include "dali/operators/image/convolution/filter.h"
#include "dali/operators/image/convolution/filter_gpu.h"

namespace dali {

namespace filter {

extern template std::unique_ptr<OpImplBase<GPUBackend>>
get_filter_gpu_op_impl<uint8_t, uint8_t, float>(const OpSpec&, const InputDesc&);
extern template std::unique_ptr<OpImplBase<GPUBackend>>
get_filter_gpu_op_impl<float, uint8_t, float>(const OpSpec&, const InputDesc&);

extern template std::unique_ptr<OpImplBase<GPUBackend>>
get_filter_gpu_op_impl<int8_t, int8_t, float>(const OpSpec&, const InputDesc&);
extern template std::unique_ptr<OpImplBase<GPUBackend>>
get_filter_gpu_op_impl<float, int8_t, float>(const OpSpec&, const InputDesc&);

extern template std::unique_ptr<OpImplBase<GPUBackend>>
get_filter_gpu_op_impl<uint16_t, uint16_t, float>(const OpSpec&, const InputDesc&);
extern template std::unique_ptr<OpImplBase<GPUBackend>>
get_filter_gpu_op_impl<float, uint16_t, float>(const OpSpec&, const InputDesc&);

extern template std::unique_ptr<OpImplBase<GPUBackend>>
get_filter_gpu_op_impl<int16_t, int16_t, float>(const OpSpec&, const InputDesc&);
extern template std::unique_ptr<OpImplBase<GPUBackend>>
get_filter_gpu_op_impl<float, int16_t, float>(const OpSpec&, const InputDesc&);

extern template std::unique_ptr<OpImplBase<GPUBackend>>
get_filter_gpu_op_impl<uint32_t, uint32_t, float>(const OpSpec&, const InputDesc&);
extern template std::unique_ptr<OpImplBase<GPUBackend>>
get_filter_gpu_op_impl<float, uint32_t, float>(const OpSpec&, const InputDesc&);

extern template std::unique_ptr<OpImplBase<GPUBackend>>
get_filter_gpu_op_impl<int32_t, int32_t, float>(const OpSpec&, const InputDesc&);
extern template std::unique_ptr<OpImplBase<GPUBackend>>
get_filter_gpu_op_impl<float, int32_t, float>(const OpSpec&, const InputDesc&);

extern template std::unique_ptr<OpImplBase<GPUBackend>>
get_filter_gpu_op_impl<float16, float16, float>(const OpSpec&, const InputDesc&);
extern template std::unique_ptr<OpImplBase<GPUBackend>>
get_filter_gpu_op_impl<float, float16, float>(const OpSpec&, const InputDesc&);

extern template std::unique_ptr<OpImplBase<GPUBackend>> get_filter_gpu_op_impl<float, float, float>(
    const OpSpec&, const InputDesc&);

}  // namespace filter

// Passing to the kernel less samples (not split into frames) speeds-up
// the processing, so expand frames dim only if some argument was specified per-frame
template <>
bool Filter<GPUBackend>::ShouldExpand(const Workspace& ws) {
  return SequenceOperator<GPUBackend>::ShouldExpand(ws) &&
         (HasPerFramePositionalArgs(ws) || HasPerFrameArgInputs(ws));
}

template <>
template <typename Out, typename In, typename W>
std::unique_ptr<OpImplBase<GPUBackend>> Filter<GPUBackend>::GetFilterImpl<Out, In, W>(
    const OpSpec& spec, const filter::InputDesc& input_desc) {
  return filter::get_filter_gpu_op_impl<Out, In, W>(spec, input_desc);
}

DALI_REGISTER_OPERATOR(experimental__Filter, Filter<GPUBackend>, GPU);

}  // namespace dali
