// Copyright (c) 2021-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <utility>

namespace dali {

bool WebdatasetReader::SetupImpl(std::vector<OutputDesc>& output_desc, const Workspace &ws) {
  DataReader<CPUBackend, std::vector<Tensor<CPUBackend>>>::SetupImpl(output_desc, ws);
  int num_outputs = ws.NumOutput();
  int num_samples = GetCurrBatchSize();

  output_desc.resize(num_outputs);
  for (int output_idx = 0; output_idx < num_outputs; output_idx++) {
    output_desc[output_idx].shape = TensorListShape<>(num_samples, 1);
  }

  for (int data_idx = 0; data_idx < num_samples; data_idx++) {
    auto& sample = GetSample(data_idx);
    for (int output_idx = 0; output_idx < num_outputs; output_idx++) {
      output_desc[output_idx].shape.set_tensor_shape(data_idx, sample[output_idx].shape());
      output_desc[output_idx].type = sample[output_idx].type();
    }
  }
  return true;
}

void WebdatasetReader::RunImpl(Workspace &ws) {
  int num_outputs = ws.NumOutput();
  int num_samples = GetCurrBatchSize();

  bool threaded = ws.GetThreadPool().NumThreads() > 1;

  for (int output_idx = 0; output_idx < num_outputs; output_idx++) {
    auto& output = ws.Output<CPUBackend>(output_idx);
    for (int data_idx = 0; data_idx < num_samples; data_idx++) {
      auto& sample = GetSample(data_idx);
      ThreadPool::Work copy_task = [output_idx = output_idx, data_idx = data_idx, &output,
                                    &sample](int) {
        output.SetMeta(data_idx, sample[output_idx].GetMeta());
        std::memcpy(output.raw_mutable_tensor(data_idx), sample[output_idx].raw_data(),
                    sample[output_idx].nbytes());
      };
      if (threaded) {
        ws.GetThreadPool().AddWork(std::move(copy_task), -data_idx);
      } else {
        copy_task(0);
      }
    }
  }
  if (threaded) {
    ws.GetThreadPool().RunAll();
  }
}

DALI_SCHEMA(readers__Webdataset)
    .DocStr((std::string) R"code(A reader for the webdataset format.

The webdataset format is a way of providing efficient access to datasets stored in tar archives.

Storing data in POSIX tar archives greatly speeds up I/O operations on mechanical storage devices
and on network file systems because it allows the operating system to reduce the number of I/O
operations and to read the data ahead.

WebDataset fulfils a similar function to Tensorflow's TFRecord/tf.Example classes, but is much
easier to adopt because it does not actually require any data conversion. The data is stored in
exactly the same format inside tar files as it is on disk, and all preprocessing and data
augmentation code remains unchanged.

The dataset consists of one or more tar archives, each of which is further split into samples.
A sample contains one or more components that correspond to the actual files contained within
the archive. The components that belong to a specific sample are aggregated by filename without
extension (for the specifics about the extensions please read the description of the ``ext`` parameter
below). Note that samples with their filename starting with a dot will not be loaded, as well as
entries that are not regular files.

In addition to the tar archive with data, each archive should come with a corresponding index file.
The index file can be generated using a dedicated script::

    <path_to_dali>/tools/wds2idx.py <path_to_archive> <path_to_index_file>

If the index file is not provided, it will be automatically inferred from the tar file.
Keep in mind though that it will add considerable startup time for big datasets.

The format of the index file is::

    )code" +
    detail::wds::kCurrentIndexVersion + R"code( <num_samples>
    <component1_ext> <component1_data_offset> <component1_size> <component2_ext> <component2_data_offset> <component2_size> ...
    ...


Based on https://github.com/webdataset/webdataset)code")
    .NumInput(0)
    .OutputFn([](const OpSpec& spec) {
      return spec.HasArgument("ext") ? spec.GetRepeatedArgument<std::string>("ext").size() : 0;
    })
    .AddArg("paths", R"code(The list of (one or more) paths to the webdataset archives.

Has to be the same length as the ``index_paths`` argument.)code",
            DALI_STRING_VEC)
    .AddArg("ext", R"code(The extension sets for each of the outputs produced.

The number of extension sets determines the number of outputs of the reader.
The extensions of the components are counted as the text after the first dot in the name of the file
(excluding the samples starting with a dot). The different extension options should be separated
with a semicolon (';') and may contain dots.

Example: "left.png;right.jpg")code",
            DALI_STRING_VEC)
    .AddOptionalArg("case_sensitive_extensions",
      R"code(Determines whether the extensions provided via the `ext` should be case sensitive.

Allows mixing case sizes in the `ext` argument as well as in the webdataset container. For example
when turned off: jpg, JPG, jPG should work.

If the extension characters cannot be represented as ASCI the result of turing this option off
is undefined.
)code", true)
    .AddOptionalArg("index_paths",
            R"code(The list of the index files corresponding to the respective webdataset archives.

Has to be the same length as the ``paths`` argument. In case it is not provided,
it will be inferred automatically from the webdataset archive.)code",
            std::vector<std::string>())
    .AddOptionalArg(
        "missing_component_behavior",
        R"code(Specifies what to do in case there is not any file in a sample corresponding to a certain output.

Possible behaviors:
  - "empty" (default) - in that case the output that was not set will just contain an empty tensor
  - "skip" - in that case the entire sample will just be skipped (no penalty to performance except for reduced caching of the archive)
  - "error" - in that case an exception will be raised and te execution stops)code",
        "")
    .AddOptionalArg("dtypes", R"code(Data types of the respective outputs.

The default output data types are UINT8. However, if set, each output data type should be specified.
Moreover, the tar file should be constructed so that it will only output a sample with its byte size
divisible by the size of the data type.)code",
                    DALI_DATA_TYPE_VEC,
                    nullptr)  // default is a vector of uint8
    .AddParent("LoaderBase");

DALI_REGISTER_OPERATOR(readers__Webdataset, WebdatasetReader, CPU);

}  // namespace dali
