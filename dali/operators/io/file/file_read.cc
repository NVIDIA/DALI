// Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/pipeline/operator/checkpointing/stateless_operator.h"
#include "dali/core/util.h"
#include "dali/util/file.h"

namespace dali {
namespace io {

class FileRead : public StatelessOperator<CPUBackend> {
 public:
  explicit FileRead(const OpSpec &spec)
      : StatelessOperator<CPUBackend>(spec),
        dont_use_mmap_(spec.GetArgument<bool>("dont_use_mmap")),
        use_o_direct_(spec.GetArgument<bool>("use_o_direct")) {}


  bool SetupImpl(std::vector<OutputDesc> &output_descs, const Workspace &ws) override {
    const auto &filepaths = ws.Input<CPUBackend>(0);
    DALI_ENFORCE(filepaths.type() == DALI_UINT8 || filepaths.type() == DALI_INT8,
                 "Unexpected input data type");
    auto nsamples = filepaths.shape().num_samples();
    output_descs.resize(1);
    output_descs[0].type = DALI_UINT8;
    output_descs[0].shape.resize(nsamples, 1);
    filenames_.resize(nsamples);
    files_.resize(nsamples);
    auto &tp = ws.GetThreadPool();
    FileStream::Options opts;
    opts.use_mmap = !dont_use_mmap_;
    opts.use_odirect = use_o_direct_;
    opts.read_ahead = false;
    for (int i = 0; i < nsamples; i++) {
      size_t filename_len = filepaths.tensor_shape_span(i)[0];
      filenames_[i].resize(filename_len + 1);
      std::memcpy(&filenames_[i][0], filepaths.raw_tensor(i), filename_len);
      filenames_[i][filename_len] = '\0';
      tp.AddWork(
        [&, i](int tid) {
          files_[i] = FileStream::Open(filenames_[i], opts);
          output_descs[0].shape.tensor_shape_span(i)[0] = files_[i]->Size();
        }, -i);  // FIFO order
    }
    tp.RunAll();
    return true;
  }

  void RunImpl(Workspace &ws) override {
    auto &file_contents = ws.Output<CPUBackend>(0);
    auto &tp = ws.GetThreadPool();
    for (size_t idx = 0; idx < filenames_.size(); idx++) {
      size_t file_size = file_contents.shape().tensor_size(idx);
      tp.AddWork(
          [&, file_size, idx](int tid) {
            auto &stream = files_[idx];
            auto read_nbytes =
                stream->Read(file_contents.mutable_tensor<uint8_t>(idx), file_size);
            DALI_ENFORCE(read_nbytes == file_size,
                         make_string("Expected to read ", file_size, " bytes, but got only ",
                                     read_nbytes, " bytes."));
            stream->Close();
            stream.reset();
          },
          file_size);
    }
    tp.RunAll();
  }

 private:
  bool dont_use_mmap_;
  bool use_o_direct_;
  std::vector<std::string> filenames_;
  std::vector<std::unique_ptr<FileStream>> files_;
};

}  // namespace io

DALI_REGISTER_OPERATOR(io__file__Read, io::FileRead, CPU);
DALI_SCHEMA(io__file__Read)
    .DocStr(R"(Reads raw file contents from an encoded filename represented by a 1D byte array.

.. note::
  To produce a compatible encoded filepath from Python (e.g. in an external_source node generator),
  use `np.frombuffer(filepath_str.encode("utf-8"), dtype=types.UINT8)`.
)")
    .NumOutput(1)
    .NumInput(1)
    .InputDox(0, "filepaths", "TensorList", "File paths to read from.")
    .AddOptionalArg(
        "dont_use_mmap",
        R"code(If set to True, it will use plain file I/O instead of trying to map the file into memory.

Mapping provides a small performance benefit when accessing a local file system, but for most network file
systems, it does not provide a benefit)code",
        false)
    .AddOptionalArg(
        "use_o_direct",
        R"code(If set to True, the data will be read directly from the storage bypassing system
cache.

Mutually exclusive with ``dont_use_mmap=False``.)code",
        false);

}  // namespace dali
