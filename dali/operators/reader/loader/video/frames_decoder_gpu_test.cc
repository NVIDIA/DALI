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

#include <exception>

#include "dali/test/dali_test_config.h"
#include "dali/core/error_handling.h"
#include "dali/operators/reader/loader/video/video_test_base.h"
#include "dali/operators/reader/loader/video/frames_decoder_gpu.h"
#include "dali/core/dynlink_cuda.h"
#include "dali/core/cuda_utils.h"
#include "dali/core/unique_handle.h"


namespace dali {
class FramesDecoderGpuTest : public VideoTestBase {
};

struct CUDAContext : UniqueHandle<CUcontext, CUDAContext> {
  DALI_INHERIT_UNIQUE_HANDLE(CUcontext, CUDAContext);
  static CUDAContext Create(int flags, CUdevice dev) {
    CUcontext ctx;
    CUDA_CALL(cuCtxCreate(&ctx, 0, dev));
    return CUDAContext(ctx);
  }

  static void DestroyHandle(CUcontext ctx) {
    CUDA_DTOR_CALL(cuCtxDestroy(ctx));
  }
};


// TEST_F(FramesDecoderGpuTest, VariableFrameRateAvi) {
//     ASSERT_TRUE(cuInitChecked());
//     CUdevice cu_test_device = 0;
//     CUDA_CALL(cuDeviceGet(&cu_test_device, 0));
//     auto cu_test_ctx = CUDAContext::Create(0, cu_test_device);
//     CUDA_CALL(cuCtxSetCurrent(cu_test_ctx));

//     std::string path = testing::dali_extra_path() + "/db/video/cfr/test_1.avi";

//     // Create file, build index
//     FramesDecoderGpu file(path);

//     ASSERT_EQ(file.Height(), 720);
//     ASSERT_EQ(file.Width(), 1280);
//     ASSERT_EQ(file.Channels(), 3);
//     ASSERT_EQ(file.NumFrames(), 50);

//     std::vector<uint8_t> frame(file.FrameSize());

//     // Read first frame
//     for (int i = 0; i < 10; ++i)
//         file.ReadNextFrame(frame.data());
//     // this->CompareFrames(frame.data(), this->GetCfrFrame(0, 0), file.FrameSize(), 50);

//     // // Seek to frame
//     // file.SeekFrame(25);
//     // file.ReadNextFrame(frame.data());
//     // this->CompareFrames(frame.data(), this->GetCfrFrame(0, 25), file.FrameSize(), 50);

//     // // Seek back to frame
//     // file.SeekFrame(12);
//     // file.ReadNextFrame(frame.data());
//     // this->CompareFrames(frame.data(), this->GetCfrFrame(0, 12), file.FrameSize(), 50);

//     // // Seek to last frame (flush frame)
//     // file.SeekFrame(49);
//     // file.ReadNextFrame(frame.data());
//     // this->CompareFrames(frame.data(), this->GetCfrFrame(0, 49), file.FrameSize(), 50);

//     // // Wrap around to first frame
//     // ASSERT_FALSE(file.ReadNextFrame(frame.data()));
//     // file.Reset();
//     // file.ReadNextFrame(frame.data());
//     // this->CompareFrames(frame.data(), this->GetCfrFrame(0, 0), file.FrameSize(), 50);
// }

}  // namespace dali
