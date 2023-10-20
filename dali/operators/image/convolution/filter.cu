// Copyright (c) 2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#define EXTERN_FILTER_SPECIALIZATION(OUT, IN)                                                     \
  extern template std::unique_ptr<OpImplBase<GPUBackend>> get_filter_gpu_op_impl<OUT, IN, float>( \
      const OpSpec&, const InputDesc&);

EXTERN_FILTER_SPECIALIZATION(uint8_t, uint8_t)
EXTERN_FILTER_SPECIALIZATION(float, uint8_t)
EXTERN_FILTER_SPECIALIZATION(int8_t, int8_t)
EXTERN_FILTER_SPECIALIZATION(float, int8_t)
EXTERN_FILTER_SPECIALIZATION(uint16_t, uint16_t)
EXTERN_FILTER_SPECIALIZATION(float, uint16_t)
EXTERN_FILTER_SPECIALIZATION(int16_t, int16_t)
EXTERN_FILTER_SPECIALIZATION(float, int16_t)
EXTERN_FILTER_SPECIALIZATION(float16, float16)
EXTERN_FILTER_SPECIALIZATION(float, float16)
EXTERN_FILTER_SPECIALIZATION(float, float)

}  // namespace filter

template <>
template <typename Out, typename In, typename W>
std::unique_ptr<OpImplBase<GPUBackend>> Filter<GPUBackend>::GetFilterImpl(
    const OpSpec& spec, const filter::InputDesc& input_desc) {
  return filter::get_filter_gpu_op_impl<Out, In, W>(spec, input_desc);
}

DALI_REGISTER_OPERATOR(experimental__Filter, Filter<GPUBackend>, GPU);

}  // namespace dali
