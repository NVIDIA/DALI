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
#include <iostream>
#include <set>
#include <stdexcept>
#include <vector>
#include "dali/imgcodec/decoders/decoder_parallel_impl.h"

namespace dali::imgcodec {

struct DummyDecoder : public BatchParallelDecoderImpl {
 public:
  DummyDecoder() : BatchParallelDecoderImpl(CPU_ONLY_DEVICE_ID, {}) {}
  std::atomic_int counter{0};
  std::vector<std::exception_ptr> error_to_throw;
  cspan<ImageSource *> sources;
  span<SampleView<CPUBackend>> outputs;
  cspan<ROI> rois;
  vector<int> seen;
  DecodeParams opts;

  DecodeResult DecodeImplTask(int thread_idx,
                              SampleView<CPUBackend> out,
                              ImageSource *in,
                              DecodeParams opts,
                              const ROI &roi) override {
    counter++;

    std::string message;
    auto error = [&](const std::string &msg) {
      std::cerr << msg << std::endl;
      if (!message.empty())
        message += "\n";
      message += msg;
    };

    if (opts.dtype != this->opts.dtype ||
        opts.format != this->opts.format ||
        opts.planar != this->opts.planar ||
        opts.use_orientation != this->opts.use_orientation)
      error("The decoding options not propagated correctly");

    auto it = std::find(sources.begin(), sources.end(), in);
    if (it == sources.end())
      error("The source given to DecodeImplTask not seen in the full work.");

    int sample_idx = it - sources.begin();
    if (out.raw_data() != outputs[sample_idx].raw_data())
      error("The index inferred from source doesn't point to the correct output.");
    if (rois.empty() && (!roi.begin.empty() || !roi.end.empty()))
      error("Found a non-empty per-sample ROI when no ROIs were given.");
    else if (!rois.empty() &&
             (roi.begin != rois[sample_idx].begin || roi.end != rois[sample_idx].end))
      error("The index inferred from source doesn't point to the correct ROI.");

    if (seen[sample_idx])
      error("The same sample seen more than once.");

    seen[sample_idx] = true;

    if (!message.empty())
      throw std::logic_error(message);

    if (error_to_throw[sample_idx])
      return DecodeResult::Failure(error_to_throw[sample_idx]);
    else
      return DecodeResult::Success();
  }
};

class DecoderParallelImplTest : public ::testing::Test,
                                public ::testing::WithParamInterface<bool> {
};

TEST_P(DecoderParallelImplTest, ScheduleDecode) {
  bool use_roi = GetParam();

  ThreadPool tp(4, CPU_ONLY_DEVICE_ID, false, "DecoderParallelImplTest");
  DecodeContext ctx(&tp, 0);
  DummyDecoder dd;
  const int kBatchSize = 16;
  std::set<int> error_indices = { 1, 5, 13 };

  SampleView<CPUBackend> sv[kBatchSize];
  ImageSource srcs[kBatchSize];
  ImageSource *psrc[kBatchSize];
  for (int i = 0; i < kBatchSize; i++) {
    void *data = &sv[i];  // just some pointer that will be unique within the batch
    sv[i] = SampleView<CPUBackend>(data, { 1 }, DALI_INT32);

    psrc[i] = &srcs[i];
  }

  vector<ROI> rois;
  if (use_roi) {
    rois.resize(kBatchSize);
    for (int i = 0; i < kBatchSize; i++) {
      rois[i].begin = { i + 1, i + 2 };
      rois[i].end =   { i + 3, i + 4 };
    }
  }

  std::exception_ptr exception = std::make_exception_ptr(std::runtime_error("some_error"));
  dd.error_to_throw.resize(kBatchSize);
  for (int idx : error_indices)
    dd.error_to_throw[idx] = exception;

  // put some non-defaults in the opts
  dd.opts.dtype = DALI_INT32;
  dd.opts.planar = true;

  dd.rois = make_span(rois);
  dd.outputs = make_span(sv);
  dd.sources = make_span(psrc);

  dd.seen.resize(kBatchSize);

  FutureDecodeResults f = dd.ScheduleDecode(ctx, dd.outputs, dd.sources, dd.opts, dd.rois);
  auto res = f.get_all_ref();
  EXPECT_EQ(dd.counter, kBatchSize);
  for (int i = 0; i < kBatchSize; i++) {
    bool expect_error = error_indices.count(i);
    EXPECT_EQ(res[i].success, !expect_error);
    if (expect_error) {
      EXPECT_EQ(res[i].exception, exception);
      if (res[i].exception != exception) {
        EXPECT_NO_THROW(std::rethrow_exception(res[i].exception));
      }
    } else if (res[i].exception) {
      EXPECT_NO_THROW(std::rethrow_exception(res[i].exception));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(ROI, DecoderParallelImplTest, ::testing::Values(false, true));

}  // namespace dali::imgcodec
