// Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
namespace detail {

inline int FitsReaderOutputFn(const OpSpec &spec) {
  // there is default value provided, so no checking if arg exists is needed
  return spec.GetRepeatedArgument<int>("hdu_indices").size();
}

}  // namespace detail

DALI_REGISTER_OPERATOR(experimental__readers__Fits, FitsReaderCPU, CPU);

DALI_SCHEMA(experimental__readers__Fits)
    .DocStr(R"(Reads Fits image HDUs from a directory.

This operator can be used in the following modes:

1. Read all files from a directory indicated by `file_root` that match given `file_filter`.
2. Read file names from a text file indicated in `file_list` argument.
3. Read files listed in `files` argument.
4. Number of outputs per sample corresponds to the length of `hdu_indices` argument. By default,
first HDU with data is read from each file, so the number of outputs defaults to 1. 
)")
    .NumInput(0)
    .OutputFn(detail::FitsReaderOutputFn)
    .AddOptionalArg<string>("file_root",
                            R"(Path to a directory that contains the data files.

If not using `file_list` or `files`. this directory is traversed to discover the files.
`file_root` is required in this mode of operation.)",
                            nullptr)
    .AddOptionalArg(
        "file_filter",
        R"(If a value is specified, the string is interpreted as glob string to filter the
list of files in the sub-directories of the `file_root`.

This argument is ignored when file paths are taken from `file_list` or `files`.)",
        "*.fits")
    .AddOptionalArg<string>("file_list",
                            R"(Path to a text file that contains filenames (one per line).
The filenames are relative to the location of the text file or to `file_root`, if specified.

This argument is mutually exclusive with `files`.)",
                            nullptr)
    .AddOptionalArg("shuffle_after_epoch",
                    R"(If set to True, the reader shuffles the entire dataset after each epoch.

`stick_to_shard` and `random_shuffle` cannot be used when this argument is set to True.)",
                    false)
    .AddOptionalArg<vector<string>>("files", R"(A list of file paths to read the data from.

If `file_root` is provided, the paths are treated as being relative to it.

This argument is mutually exclusive with `file_list`.)",
                                    nullptr)
    .AddOptionalArg("hdu_indices",
                    R"(HDU indices to read. If not provided, the first HDU after the primary 
will be yielded. Since HDUs are indexed starting from 1, the default value is as follows: hdu_indices = [2].
Size of the provided list hdu_indices defines number of outputs per sample.)",
                    std::vector<int>{2})
    .AddOptionalArg("dtypes", R"code(Data types of the respective outputs.

If specified, it must be a list of types of respective outputs. By default, all outputs are assumed to be UINT8.")code",
                    DALI_DATA_TYPE_VEC,
                    nullptr)  // default is a vector of uint8
    .AddParent("LoaderBase");

void FitsReaderCPU::RunImpl(Workspace &ws) {
  int num_outputs = ws.NumOutput();
  int num_samples = GetCurrBatchSize();

  bool threaded = ws.GetThreadPool().NumThreads() > 1;
  for (int output_idx = 0; output_idx < num_outputs; output_idx++) {
    auto &output = ws.Output<CPUBackend>(output_idx);
    for (int file_idx = 0; file_idx < num_samples; file_idx++) {
      auto &sample = GetSample(file_idx);
      ThreadPool::Work copy_task = [output_idx = output_idx, data_idx = file_idx, &output,
                                    &sample](int) {
        std::memcpy(output.raw_mutable_tensor(data_idx), sample.data[output_idx].raw_data(),
                    sample.data[output_idx].nbytes());
      };
      if (threaded) {
        ws.GetThreadPool().AddWork(std::move(copy_task), -file_idx);
      } else {
        copy_task(0);
      }
    }
  }
  if (threaded) {
    ws.GetThreadPool().RunAll();
  }
}

}  // namespace dali
