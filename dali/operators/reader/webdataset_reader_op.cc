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

#include "dali/operators/reader/webdataset_reader_op.h"
#include <algorithm>
#include <cstring>
#include <string>

namespace dali {

void WebdatasetReader::RunImpl(HostWorkspace& ws) {
  int num_outputs = ws.NumOutput();
  int num_samples = GetCurrBatchSize();

  for (int data_idx = 0; data_idx < num_samples; data_idx++) {
    auto& sample = GetSample(data_idx);
    for (int output_idx = 0; output_idx < num_outputs; output_idx++) {
      ws.OutputRef<CPUBackend>(output_idx)[data_idx].Resize(sample[output_idx].shape());
      ws.OutputRef<CPUBackend>(output_idx)[data_idx].set_type(sample[output_idx].type());
      ws.OutputRef<CPUBackend>(output_idx)[data_idx].SetMeta(sample[output_idx].GetMeta());
      std::memcpy(ws.OutputRef<CPUBackend>(output_idx)[data_idx].raw_mutable_data(),
                  sample[output_idx].raw_data(), sample[output_idx].nbytes());
    }
  }
}

DALI_SCHEMA(readers__Webdataset)
    .DocStr(
        R"code(A reader for the webdataset format.

The webdataset format is a dataset meant to improve caching of the data between the samples read.
The data itself is contained within one or several tar archives, each of which is further split into
samples, each of which contains one or several components that correspond to the actual file
contained within the archive. The components that correspond to specific sample are aggregated by
the part of the filepath that does not correspond to the extension (for the specifics about the
extensions please read the description of the ``ext`` parameter below). Note that samples with
their filename starting with a dot will not be loaded, as well as entries that are not a file.

In addition to the tar archive with data, each archive should come with a corresponding index file.
The format of the index file is as follows:
<offset behind the contents of the last archive entry> <number_of_samples>
<sample1_start_offset> <component1_ext> <component1_size> <component2_ext> <component2_size> ...
<sample2_start_offset> <component1_ext> <component1_size> <component2_ext> <component2_size> ...
...)code")
    .NumInput(0)
    .OutputFn([](const OpSpec& spec) {
      return spec.HasArgument("ext") ? spec.GetRepeatedArgument<std::string>("ext").size() : 0;
    })
    .AddArg("uris", R"code(The list of (one or more) paths to the webdataset archives.
Has to be the same length as the ``index_paths`` argument.)code",
            DALI_STRING_VEC)
    .AddArg("index_paths",
            R"code(The list of the index files corresponding to the respective webdataset archives.
Has to be the same length as the ``uris`` argument.)code",
            DALI_STRING_VEC)
    .AddArg("ext", R"code(The extension sets for each of the outputs produced.
The number of extension sets determines the number of outputs of the reader.
The extensions of the components are counted as the text after the first dot in the name of the file 
(excluding the samples starting with a dot). The different extension options should be separated
with a semicolon (';') and may contain dots.

Example: "left.png;right.jpg")code",
            DALI_STRING_VEC)
    .AddOptionalArg(
        "missing_component_behavior",
        R"code(Specifies what to do in case there is not any file corresponding to a certain output.
Three behaviors are possible (case-insensitive):
  - "empty" (default) - in that case the output that was not set will just contain an empty tensor
  - "skip" - in that case the entire sample will just be skipped (no penalty to performance except
             for reduced caching of the archive)
  - "error" - in that case an exception will be raised)code",
        "")
    .AddOptionalArg("dtypes", R"code(Data types of the respective outputs.
The default output data types are INT8. However, if set, each output data type should be specified.
Moreover, the tar file should be constructed so that it will only output a sample with its byte size
divisible by the size of the data type. )code",
                    DALI_DATA_TYPE_VEC,
                    nullptr)  // default is a vector of uint8
    .AddParent("LoaderBase");

DALI_REGISTER_OPERATOR(readers__Webdataset, WebdatasetReader, CPU);

}  // namespace dali
