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

#ifndef DALI_KERNELS_SIGNAL_DCT_DCT_GPU_H_
#define DALI_KERNELS_SIGNAL_DCT_DCT_GPU_H_

#include <memory>
#include <vector>
#include <map>
#include <utility>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/core/format.h"
#include "dali/core/util.h"
#include "dali/kernels/kernel.h"
#include "dali/kernels/signal/dct/dct_args.h"
#include "dali/kernels/common/block_setup.h"
#include "dali/core/cuda_event.h"

namespace dali {
namespace kernels {
namespace signal {
namespace dct {

class BlockSetupInner {
 public:
  struct BlockDesc {
    int64_t sample_idx;
    int64_t frame_start;
    int64_t frame_count;
  };

  void Setup(const TensorListShape<3> &reduced_shape);

  const std::vector<BlockDesc> &Blocks() {
    return blocks_;
  }

  dim3 BlockDim() {
    return dim3(32, 8);
  }

  dim3 GridDim() {
    return dim3(blocks_.size());
  }

  template <typename OutputType, typename InputType>
  size_t SharedMemSize(int64_t max_input_length, int64_t max_cos_table_size) {
    return sizeof(InputType) * max_input_length * frames_per_block_
           + sizeof(OutputType) * max_cos_table_size;
  }

 private:
  std::vector<BlockDesc> blocks_{};
  const int64_t frames_per_block_ = 8;
};

/**
 * @brief Discrete Cosine Transform 1D GPU kernel.
 *        Performs a DCT transformation over a single dimension in a multi-dimensional input.
 *
 * @remarks It supports DCT types I, II, III and IV decribed here:
 *          https://en.wikipedia.org/wiki/Discrete_cosine_transform
 *          DCT generally stands for type II and inverse DCT stands for DCT type III
 *
 * @see DCTArgs
 */
template <typename OutputType = float,  typename InputType = OutputType>
class DLL_PUBLIC Dct1DGpu {
 public:
  struct SampleDesc {
    OutputType *output;
    const InputType *input;
    const OutputType *cos_table;
    ivec3 in_stride;
    ivec3 out_stride;
    int input_length;
  };

 private:
  /// @brief Calculate the output shape, reduced to 3D
  static TensorShape<3> reduce_shape(span<const int64_t> shape, int axis, int ndct = -1) {
    assert(axis < shape.size());
    auto outer_dim = volume(shape.begin(), shape.begin() + axis);
    auto inner_dim = volume(shape.begin() + axis + 1, shape.end());
    if (ndct >= 0)
      return {outer_dim, ndct, inner_dim};
    else
      return {outer_dim, shape[axis], inner_dim};
  }

 public:
  static_assert(std::is_floating_point<InputType>::value,
    "Data type should be floating point");
  static_assert(std::is_same<OutputType, InputType>::value,
    "Data type conversion is not supported");

  DLL_PUBLIC Dct1DGpu(): buffer_events_{CUDAEvent::Create(), CUDAEvent::Create()} {};

  DLL_PUBLIC KernelRequirements Setup(KernelContext &context,
                                      const InListGPU<InputType> &in,
                                      span<const DctArgs> args, int axis);

  DLL_PUBLIC void Run(KernelContext &context,
                      const OutListGPU<OutputType> &out,
                      const InListGPU<InputType> &in,
                      InTensorGPU<float, 1> lifter_coeffs);

 private:
  void RunInnerDCT(KernelContext &context, int64_t max_input_length,
                   InTensorGPU<float, 1> lifter_coeffs);

  void RunPlanarDCT(KernelContext &context, int max_ndct,
                    InTensorGPU<float, 1> lifter_coeffs);

  std::map<std::pair<int, DctArgs>, OutputType*> cos_tables_{};
  std::vector<DctArgs> args_{};
  BlockSetup<3, -1> block_setup_{};
  BlockSetupInner block_setup_inner_{};
  std::vector<SampleDesc> sample_descs_{};
  int64_t max_cos_table_size_ = 0;
  int axis_ = -1;
  bool inner_axis_ = false;
  CUDAEvent buffer_events_[2];
};

}  // namespace dct
}  // namespace signal
}  // namespace kernels
}  // namespace dali

#endif  // DALI_KERNELS_SIGNAL_DCT_DCT_GPU_H_
