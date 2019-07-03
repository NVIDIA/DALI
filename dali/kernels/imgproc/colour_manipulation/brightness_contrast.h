// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_BRIGHTNESS_CONTRAST_CPU_H
#define DALI_BRIGHTNESS_CONTRAST_CPU_H

#include "dali/util/ocv.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/kernel.h"
#include "dali/pipeline/data/types.h"

namespace dali {
namespace kernels {

template<typename ComputeBackend, typename InputType, typename OutputType>
class DLL_PUBLIC BrightnessContrastCPU {
 private:
  using StorageBackend = compute_to_storage_t<ComputeBackend>;
 public:

  DLL_PUBLIC KernelRequirements
  Setup(KernelContext &context, const InTensor<StorageBackend, InputType, 3> &image) {
    KernelRequirements req;
    req.output_shapes = {TensorListShape<DynamicDimensions>({image.shape})};
    return req;
  }


  /**
   * @param out Assumes, that memory is already allocated
   * @param brightness Additive brightness delta. 0 denotes no change
   * @param contrast Multiplicative contrast delta. 1 denotes no change
   */
  DLL_PUBLIC void Run(KernelContext &context, const InTensor<StorageBackend, InputType, 3> &in,
                      OutTensor<StorageBackend, OutputType, 3> &out, InputType brightness,
                      InputType contrast) {
    for (int i = 0; i < out.num_elements(); i++) {
      out.data[i] = in.data[i] * contrast + brightness;
    }
  }
};

}  // namespace kernels
}  // namespace dali

#endif //DALI_BRIGHTNESS_CONTRAST_CPU_H
