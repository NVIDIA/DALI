// Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
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

#include "dali/kernels/transpose/transpose.h"
#include "dali/core/static_switch.h"
#include "dali/operators/reader/numpy_reader_op.h"

namespace dali {

#define NUMPY_ALLOWED_TYPES \
  (bool, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, float16, \
  double)

void NumpyReader::TransposeHelper(Tensor<CPUBackend>& output, const Tensor<CPUBackend>& input) {
  auto& in_shape = input.shape();
  auto& out_shape = output.shape();
  auto& type = input.type();
  int n_dims = in_shape.sample_dim();
  SmallVector<int, 6> perm;
  perm.resize(n_dims);
  for (int i = 0; i < n_dims; ++i)
    perm[i] = n_dims - i - 1;

  auto input_type = type.id();
  TYPE_SWITCH(input_type, type2id, InputType, NUMPY_ALLOWED_TYPES, (
    kernels::Transpose(
      TensorView<StorageCPU, InputType>{output.mutable_data<InputType>(), out_shape},
      TensorView<StorageCPU, const InputType>{input.data<InputType>(), in_shape},
      make_cspan(perm));
  ), DALI_FAIL(make_string("Unsupported input type: ", input_type)));  // NOLINT
}

DALI_REGISTER_OPERATOR(readers__Numpy, NumpyReader, CPU);

DALI_SCHEMA(readers__Numpy)
  .DocStr(R"(Reads Numpy arrays from a directory.

This operator can be used in the following modes:

1. Read all files from a directory indicated by ``file_root`` that match given ``file_filter``.
2. Read file names from a text file indicated in ``file_list`` argument.
3. Read files listed in ``files`` argument.

.. note::
  The ``gpu`` backend requires cuFile/GDS support (418.x driver family or newer). Please check
  the relevant GDS package for more details.
)")
  .NumInput(0)
  .NumOutput(1)  // (Arrays)
  .AddOptionalArg<string>("file_root",
      R"(Path to a directory that contains the data files.

If not using ``file_list`` or ``files``, this directory is traversed to discover the files.
``file_root`` is required in this mode of operation.)",
      nullptr)
  .AddOptionalArg("file_filter",
      R"(If a value is specified, the string is interpreted as glob string to filter the
list of files in the sub-directories of the ``file_root``.

This argument is ignored when file paths are taken from ``file_list`` or ``files``.)", "*.npy")
  .AddOptionalArg<string>("file_list",
      R"(Path to a text file that contains filenames (one per line)
where the filenames are relative to the location of that file or to ``file_root``, if specified.

This argument is mutually exclusive with ``files``.)", nullptr)
.AddOptionalArg("shuffle_after_epoch",
      R"(If set to True, the reader shuffles the entire dataset after each epoch.

``stick_to_shard`` and ``random_shuffle`` cannot be used when this argument is set to True.)",
      false)
  .AddOptionalArg<vector<string>>("files", R"(A list of file paths to read the data from.

If ``file_root`` is provided, the paths are treated as being relative to it.

This argument is mutually exclusive with ``file_list``.)", nullptr)
  .AddOptionalArg("register_buffers",
      R"code(Applies **only** to the ``gpu`` backend type.

If true, the device I/O buffers will be registered with cuFile. It is not recommended if the sample
sizes vary a lot.)code", true)
  .AddOptionalArg("cache_header_information",
      R"code(If set to True, the header information for each file is cached, improving access
speed.)code",
      false)
  .AddParent("LoaderBase");


// Deprecated alias
DALI_REGISTER_OPERATOR(NumpyReader, NumpyReader, CPU);

DALI_SCHEMA(NumpyReader)
    .DocStr("Legacy alias for :meth:`readers.numpy`.")
    .NumInput(0)
    .NumOutput(1)  // (Arrays)
    .AddParent("readers__Numpy")
    .MakeDocPartiallyHidden()
    .Deprecate(
        "readers__Numpy",
        R"code(In DALI 1.0 all readers were moved into a dedicated :mod:`~nvidia.dali.fn.readers`
submodule and renamed to follow a common pattern. This is a placeholder operator with identical
functionality to allow for backward compatibility.)code");  // Deprecated in 1.0;

bool NumpyReader::SetupImpl(std::vector<OutputDesc> &output_desc,
                            const workspace_t<CPUBackend> &ws) {
  // If necessary start prefetching thread and wait for a consumable batch
  DataReader<CPUBackend, ImageFileWrapper>::SetupImpl(output_desc, ws);

  const auto &file_0 = GetSample(0);
  TypeInfo output_type = file_0.image.type();
  int ndim = file_0.image.shape().sample_dim();

  TensorListShape<> sh(max_batch_size_, ndim);
  for (int i = 0; i < max_batch_size_; i++) {
    auto sample_sh = sh.tensor_shape_span(i);
    const auto& file_i = GetSample(i);
    const auto &file_sh = file_i.image.shape();
    if (file_i.meta == "transpose:false") {
      for (int d = 0; d < ndim; d++)
        sample_sh[d] = file_sh[d];
    } else {
      for (int d = 0; d < ndim; d++)
        sample_sh[d] = file_sh[ndim - 1 -d];
    }
    sh.set_tensor_shape(i, sample_sh);
  }
  output_desc.resize(1);
  output_desc[0].shape = std::move(sh);
  output_desc[0].type = output_type;
  return true;
}

void NumpyReader::RunImpl(HostWorkspace &ws) {
  auto &output = ws.OutputRef<CPUBackend>(0);
  const auto &out_sh = output.shape();
  auto &thread_pool = ws.GetThreadPool();
  for (int sample_idx = 0; sample_idx < max_batch_size_; sample_idx++) {
    thread_pool.AddWork([&, sample_idx](int tid) {
      const auto& file_i = GetSample(sample_idx);
      int64_t nbytes = file_i.image.nbytes();
      if (file_i.meta == "transpose:false") {
        std::memcpy(output[sample_idx].raw_mutable_data(), file_i.image.raw_data(), nbytes);
      } else {
        TransposeHelper(output[sample_idx], file_i.image);
      }
      output[sample_idx].SetSourceInfo(file_i.image.GetSourceInfo());
    }, out_sh.tensor_size(sample_idx));
  }
  thread_pool.RunAll();
}

}  // namespace dali
