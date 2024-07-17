// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/image/remap/cvcuda/matrix_adjust.h"

#include <dali/core/util.h>
#include <dali/core/geom/mat.h>
#include <nvcv/cuda/TensorWrap.hpp>

namespace dali {
namespace warp_perspective {

using MatricesWrap = nvcv::cuda::TensorWrap<float, 9 * sizeof(float), sizeof(float)>;

__global__ void adjustMatricesKernel2(MatricesWrap wrap, int batch_size) {
    // To adjust the matrix to OpenCV pixel coordinates, we need to apply operators changing
    // the coordinates system basis. We do it by multiplying the matrix on both sides
    // by the opposite translation matrices.
    // The same routine can be used regardless if the inverse_map is used or not
    // because inverting the matrix preserves the basis change.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) {
        return;
    }
    auto *matrix = reinterpret_cast<mat3*>(wrap.ptr(tid));

    mat3 shift = {{
        {1., 0., 0.5},
        {0., 1., 0.5},
        {0., 0., 1.}
    }};

    mat3 shift_back = {{
        {1., 0., -0.5},
        {0., 1., -0.5},
        {0., 0., 1.}
    }};

    *matrix = shift_back * *matrix * shift;
}

void adjustMatrices(nvcv::Tensor &matrices, cudaStream_t stream) {
    auto data = *matrices.exportData<nvcv::TensorDataStridedCuda>();
    int bs = data.shape()[0];
    MatricesWrap wrap(data);

    int num_blocks = div_ceil(bs, 1024);
    int threads_per_block = std::min(bs, 1024);
    adjustMatricesKernel2<<<num_blocks, threads_per_block, 0, stream>>>(wrap, bs);
}

}  // namespace warp_perspective
}  // namespace dali
