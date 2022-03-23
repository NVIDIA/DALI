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

#include "dali/operators/image/convolution/laplacian_gpu.h"

namespace dali {
namespace laplacian {

template op_impl_uptr GetLaplacianGpuImpl<int64_t, int64_t>(const OpSpec* spec,
                                                            const DimDesc& dim_desc);
template op_impl_uptr GetLaplacianGpuImpl<float, int64_t>(const OpSpec* spec,
                                                          const DimDesc& dim_desc);

}  // namespace laplacian
}  // namespace dali
