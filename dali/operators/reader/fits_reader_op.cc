// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <string>

#include "dali/core/backend_tags.h"
#include "dali/core/static_switch.h"
#include "dali/kernels/slice/slice_cpu.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_cpu.h"
#include "dali/kernels/transpose/transpose.h"
#include "dali/operators/reader/fits_reader_op.h"
#include "dali/pipeline/data/backend.h"

namespace dali {

static void CopyHelper(SampleView<CPUBackend> output, ConstSampleView<CPUBackend> input,
                       ThreadPool &thread_pool, int min_blk_sz, int req_nblocks) {
  auto *out_ptr = static_cast<uint8_t *>(output.raw_mutable_data());
  const auto *in_ptr = static_cast<const uint8_t *>(input.raw_data());
  auto nelements = input.shape().num_elements();
  auto nbytes = nelements * TypeTable::GetTypeInfo(input.type()).size();
  if (nelements <= min_blk_sz) {
    thread_pool.AddWork([=](int tid) { std::memcpy(out_ptr, in_ptr, nbytes); }, nelements);
  } else {
    int64_t prev_b_start = 0;
    for (int b = 0; b < req_nblocks; b++) {
      int64_t b_start = prev_b_start;
      int64_t b_end = prev_b_start = nbytes * (b + 1) / req_nblocks;
      int64_t b_size = b_end - b_start;
      thread_pool.AddWork(
          [=](int tid) { std::memcpy(out_ptr + b_start, in_ptr + b_start, b_size); }, b_size);
    }
  }
}

DALI_REGISTER_OPERATOR(experimental_readers__Fits, FitsReaderCPU, CPU);

DALI_SCHEMA(experimental_readers__Fits)
    .DocStr(R"(Reads Fits image HDUs from a directory.

This operator can be used in the following modes:

1. Read all files from a directory indicated by ``file_root`` that match given ``file_filter``.
2. Read file names from a text file indicated in ``file_list`` argument.
3. Read files listed in ``files`` argument.
)")
    .NumInput(0)
    .NumOutput(1)  // (Arrays)
    .AddOptionalArg<string>("file_root",
                            R"(Path to a directory that contains the data files.

If not using ``file_list`` or ``files``, this directory is traversed to discover the files.
``file_root`` is required in this mode of operation.)",
                            nullptr)
    .AddOptionalArg(
        "file_filter",
        R"(If a value is specified, the string is interpreted as glob string to filter the
list of files in the sub-directories of the ``file_root``.

This argument is ignored when file paths are taken from ``file_list`` or ``files``.)",
        "*.fits")
    .AddOptionalArg<string>("file_list",
                            R"(Path to a text file that contains filenames (one per line)
where the filenames are relative to the location of that file or to ``file_root``, if specified.

This argument is mutually exclusive with ``files``.)",
                            nullptr)
    .AddOptionalArg("shuffle_after_epoch",
                    R"(If set to True, the reader shuffles the entire dataset after each epoch.

``stick_to_shard`` and ``random_shuffle`` cannot be used when this argument is set to True.)",
                    false)
    .AddOptionalArg<vector<string>>("files", R"(A list of file paths to read the data from.

If ``file_root`` is provided, the paths are treated as being relative to it.

This argument is mutually exclusive with ``file_list``.)",
                                    nullptr)
    .AddParent("LoaderBase");

void FitsReaderCPU::RunImpl(Workspace &ws) {
  auto &output = ws.Output<CPUBackend>(0);
  const auto &out_sh = output.shape();
  int nsamples = out_sh.num_samples();
  auto &thread_pool = ws.GetThreadPool();
  int nthreads = thread_pool.NumThreads();

  // From 1 to 10 blocks per sample depending on the nthreads/nsamples ratio
  int blocks_per_sample = std::max(1, 10 * nthreads / nsamples);
  constexpr int kThreshold = kernels::kSliceMinBlockSize;  // smaller samples will not be subdivided

  for (int i = 0; i < nsamples; i++) {
    const auto &file_i = GetSample(i);
    const auto &file_sh = file_i.get_shape();
    int64_t sample_sz = volume(file_i.get_shape());
    auto input_sample = const_sample_view(file_i.data);
    CopyHelper(output[i], input_sample, thread_pool, kThreshold, blocks_per_sample);
  }
  thread_pool.RunAll();
}

}  // namespace dali
