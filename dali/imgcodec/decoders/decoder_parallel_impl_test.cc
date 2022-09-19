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

#include <gtest/gtest.h>
#include <atomic>
#include <stdexcept>
#include "dali/imgcodec/decoders/decoder_parallel_impl.h"

namespace dali::imgcodec {

struct DummyDecoder : public BatchParallelDecoderImpl {
 public:
  DummyDecoder() : BatchParallelDecoderImpl(CPU_ONLY_DEVICE_ID, {}) {}
  std::atomic_int counter{0};
  std::exception_ptr error_to_throw;
  DecodeResult DecodeImplTask(int thread_idx,
                              SampleView<CPUBackend> out,
                              ImageSource *in,
                              DecodeParams opts,
                              const ROI &roi) override {
    counter++;
    if (error_to_throw)
      return DecodeResult::Failure(error_to_throw);
    else
      return DecodeResult::Success();
  }
};

TEST(DecoderParallelImplTest, Success) {
  ThreadPool tp(2, CPU_ONLY_DEVICE_ID, false, "DecoderParallelImplTest");
  DecodeContext ctx(&tp, 0);
  DummyDecoder dd;
  SampleView<CPUBackend> sv[2];
  ImageSource *srcs[2] = {};
  FutureDecodeResults f = dd.ScheduleDecode(ctx, make_span(sv), make_span(srcs), {}, {});
  auto res = f.get_all_ref();
  EXPECT_EQ(dd.counter, 2);
  EXPECT_TRUE(res[0].success);
  EXPECT_TRUE(res[1].success);
}

TEST(DecoderParallelImplTest, ErrorPropagation) {
  ThreadPool tp(2, CPU_ONLY_DEVICE_ID, false, "DecoderParallelImplTest");
  DecodeContext ctx(&tp, 0);
  DummyDecoder dd;
  SampleView<CPUBackend> sv[2];
  ImageSource *srcs[2] = {};
  std::exception_ptr exception = std::make_exception_ptr(std::runtime_error("some_error"));
  dd.error_to_throw = exception;
  FutureDecodeResults f = dd.ScheduleDecode(ctx, make_span(sv), make_span(srcs), {}, {});
  auto res = f.get_all_ref();
  EXPECT_EQ(dd.counter, 2);
  EXPECT_FALSE(res[0].success);
  EXPECT_EQ(res[0].exception, exception);
  EXPECT_FALSE(res[1].success);
  EXPECT_EQ(res[1].exception, exception);
}

}  // namespace dali::imgcodec
