// Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_NVDECODER_IMGPROC_H_
#define DALI_OPERATORS_READER_NVDECODER_IMGPROC_H_


#include "dali/core/common.h"
#include "dali/operators/reader/nvdecoder/sequencewrapper.h"

namespace dali {

template<typename T>
DLL_PUBLIC void process_frame(
    cudaTextureObject_t chroma, cudaTextureObject_t luma,
    SequenceWrapper& output, int index, cudaStream_t stream,
    uint16_t input_width, uint16_t input_height,
    bool rgb, bool normalized);

}  // namespace dali

#endif  // DALI_OPERATORS_READER_NVDECODER_IMGPROC_H_
