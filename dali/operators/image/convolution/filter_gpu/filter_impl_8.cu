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

template std::unique_ptr<OpImplBase<GPUBackend>> get_filter_gpu_op_impl<uint8_t, uint8_t, float>(
    const OpSpec&, const InputLayoutDesc&);
template std::unique_ptr<OpImplBase<GPUBackend>> get_filter_gpu_op_impl<float, uint8_t, float>(
    const OpSpec&, const InputLayoutDesc&);
template std::unique_ptr<OpImplBase<GPUBackend>> get_filter_gpu_op_impl<int8_t, int8_t, float>(
    const OpSpec&, const InputLayoutDesc&);
template std::unique_ptr<OpImplBase<GPUBackend>> get_filter_gpu_op_impl<float, int8_t, float>(
    const OpSpec&, const InputLayoutDesc&);

}  // namespace filter
}  // namespace dali