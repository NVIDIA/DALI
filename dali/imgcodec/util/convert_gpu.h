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

#ifndef DALI_IMGCODEC_UTIL_CONVERT_GPU_H_
#define DALI_IMGCODEC_UTIL_CONVERT_GPU_H_

#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/data/sample_view.h"
#include "dali/imgcodec/image_decoder.h"

namespace dali {
namespace imgcodec {

/**
 * @brief Colorspace and layout conversion for GPU
 * 
 * @param out 
 * @param out_layout 
 * @param out_format 
 * @param in 
 * @param in_layout 
 * @param in_format 
 * @param stream 
 * @param roi 
 * @param multiplier 
 */
void DLL_PUBLIC Convert(
    SampleView<GPUBackend> out, TensorLayout out_layout, DALIImageType out_format,
    ConstSampleView<GPUBackend> in, TensorLayout in_layout, DALIImageType in_format,
    cudaStream_t stream, const ROI &roi, float multiplier = 1.0f);

}  // namespace imgcodec
}  // namespace dali


#endif  // DALI_IMGCODEC_UTIL_CONVERT_GPU_H_