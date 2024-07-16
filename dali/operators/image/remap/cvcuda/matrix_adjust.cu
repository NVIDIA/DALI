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

__global__ void adjustMatricesKernel(MatricesWrap wrap, int batch_size) {
    // The following operations are equivalent to the matrix
    // multiplication by transaltion operators:
    //      |1 0 -0.5|       |1 0 0.5|
    // M' = |0 1 -0.5| * M * |0 1 0.5|
    //      |0 0   1 |       |0 0  1 |
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int matrix_id = tid / 4;
    if (matrix_id >= batch_size) {
        return;
    }
    auto *data_ptr = wrap.ptr(matrix_id);
    auto *matrix = reinterpret_cast<mat3*>(data_ptr);
    int sub_tid = tid % 4;
    if (sub_tid % 2 == 0) {
        // this modifies only the first two rows
        int row_id = sub_tid / 2;
        matrix->set_row(row_id, matrix->row(row_id) - matrix->row(2) * 0.5);
    }
    __syncthreads();
    if (sub_tid < 4) {
        // this modifies only the third column
        int row_id = sub_tid;
        matrix->at(row_id, 2) = dot(matrix->row(row_id), vec3{0.5, 0.5, 1});
    }
}

void adjustMatrices(nvcv::Tensor &matrices, cudaStream_t stream) {
    auto data = *matrices.exportData<nvcv::TensorDataStridedCuda>();
    int bs = data.shape()[0];
    int num_threads = bs * 4;
    int num_blocks = div_ceil(num_threads, 32);
    int threads_per_block = std::min(32, num_threads);
    MatricesWrap wrap(data);
    adjustMatricesKernel<<<num_blocks, num_threads, 0, stream>>>(wrap, bs);
}


}  // namespace warp_perspective
}  // namespace dali
