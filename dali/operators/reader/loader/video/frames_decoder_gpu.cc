// Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/reader/loader/video/frames_decoder_gpu.h"
#include "dali/operators/reader/loader/video/nvdecode/NvDecoder.h"
#include "dali/core/error_handling.h"
#include "dali/core/cuda_utils.h"


namespace dali {

FramesDecoderGpu::FramesDecoderGpu(const std::string &filename) :
  FramesDecoder(filename) {}

bool FramesDecoderGpu::DecodeFrame(uint8_t *data, bool copy_to_output) {
  CUcontext context;
  CUDA_CALL(cuCtxGetCurrent(&context));

  Rect r = {};
  Dim d = {};

  static NvDecoder dec(context, false, cudaVideoCodec_MPEG4, false, false, &r, &d);
  int n_frames_returned = dec.Decode(av_state_->packet_->data, av_state_->packet_->size);

  return n_frames_returned > 0;
}


}  // namespace dali
