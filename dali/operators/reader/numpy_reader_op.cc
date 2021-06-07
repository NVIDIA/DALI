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

#include "dali/kernels/slice/slice_cpu.h"
#include "dali/kernels/transpose/transpose.h"
#include "dali/core/static_switch.h"
#include "dali/operators/reader/numpy_reader_op.h"

namespace dali {

#define NUMPY_ALLOWED_TYPES \
  (bool, uint8_t, uint16_t, uint32_t, uint64_t, int8_t, int16_t, int32_t, int64_t, float, float16, \
  double)

void NumpyReader::TransposeHelper(Tensor<CPUBackend> &output, const Tensor<CPUBackend> &input) {
  int n_dims = input.shape().sample_dim();
  SmallVector<int, 6> perm;
  perm.resize(n_dims);
  for (int i = 0; i < n_dims; ++i)
    perm[i] = n_dims - i - 1;
  TYPE_SWITCH(input.type().id(), type2id, T, NUMPY_ALLOWED_TYPES, (
    kernels::TransposeGrouped(view<T>(output), view<const T>(input), make_cspan(perm));
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())));  // NOLINT
}

void NumpyReader::SliceHelper(Tensor<CPUBackend> &output, const Tensor<CPUBackend> &input,
                              const CropWindow &roi) {
  int ndim = input.shape().sample_dim();
  VALUE_SWITCH(ndim, Dims, (1, 2, 3, 4, 5, 6), (
    TYPE_SWITCH(input.type().id(), type2id, T, NUMPY_ALLOWED_TYPES, (
      kernels::SliceCPU<T, T, Dims> kernel;
      kernels::SliceArgs<T, Dims> args;
      args.anchor = roi.anchor;
      args.shape = roi.shape;
      kernels::KernelContext ctx;
      auto out_view = view<T, Dims>(output);
      auto in_view = view<const T, Dims>(input);
      // no need to run Setup (we already know the output shape)
      kernel.Run(ctx, out_view, in_view, args);
    ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported number of dimensions: ", ndim)););  // NOLINT
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
    .AddOptionalArg<std::vector<int>>("roi_start",
        R"code(Start of the region-of-interest, in absolute coordinates.

This argument is incompatible with "rel_roi_start".
)code",
        nullptr, true)
    .AddOptionalArg<std::vector<float>>("rel_roi_start",
        R"code(Start of the region-of-interest, in relative coordinates (range [0.0 - 1.0]).

This argument is incompatible with "roi_start".
)code",
        nullptr, true)
    .AddOptionalArg<std::vector<int>>("roi_end",
        R"code(End of the region-of-interest, in absolute coordinates.

This argument is incompatible with "rel_roi_end", "roi_shape" and "rel_roi_shape".
)code",
        nullptr, true)
    .AddOptionalArg<std::vector<float>>("rel_roi_end",
        R"code(End of the region-of-interest, in relative coordinates (range [0.0 - 1.0]).

This argument is incompatible with "roi_end", "roi_shape" and "rel_roi_shape".
)code",
        nullptr, true)
    .AddOptionalArg<std::vector<int>>("roi_shape",
        R"code(Shape of the region-of-interest, in absolute coordinates.

This argument is incompatible with "rel_roi_shape", "roi_end" and "rel_roi_end".
)code",
        nullptr, true)
    .AddOptionalArg<std::vector<float>>("rel_roi_shape",
        R"code(Shape of the region-of-interest, in relative coordinates (range [0.0 - 1.0]).

This argument is incompatible with "roi_shape", "roi_end" and "rel_roi_end".
)code",
        nullptr, true)
    .AddOptionalArg("roi_axes",
        R"code(Order of dimensions used for the ROI anchor and shape argumens, as dimension indices.

If not provided, all the dimensions should be specified in the ROI arguments.
)code",
        std::vector<int>{})
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

  int batch_size = GetCurrBatchSize();
  const auto &file_0 = GetSample(0);
  TypeInfo output_type = file_0.image.type();
  int ndim = file_0.image.shape().sample_dim();
  TensorListShape<> sh(batch_size, ndim);

  bool need_slice = slice_attr_.template ProcessArguments<CPUBackend>(ws, batch_size, ndim);

  rois_.clear();
  if (need_slice)
    rois_.resize(batch_size);

  for (int i = 0; i < batch_size; i++) {
    const auto& file_i = GetSample(i);
    const auto &file_sh = file_i.image.shape();
    auto sample_sh = sh.tensor_shape_span(i);

    DALI_ENFORCE(
        file_i.image.shape().sample_dim() == ndim,
        make_string(
            "Inconsistent data: All samples in the batch must have the same number of dimensions. "
            "Got \"",
            file_0.filename, "\" with ", ndim, " dimensions and \"", file_i.filename, "\" with ",
            file_i.image.shape().sample_dim(), " dimensions"));
    DALI_ENFORCE(
        file_i.image.type().id() == output_type.id(),
        make_string("Inconsistent data: All samples in the batch must have the same data type. "
                    "Got \"",
                    file_0.filename, "\" with data type ", output_type.id(), " and \"",
                    file_i.filename, "\" with data type ", file_i.image.type().id()));

    bool is_transposed = !(file_i.meta == "transpose:false");
    // Calculate the full transposed shape first
    if (is_transposed) {
      for (int d = 0; d < ndim; d++)
        sample_sh[d] = file_sh[ndim - 1 - d];
    } else {
      for (int d = 0; d < ndim; d++)
        sample_sh[d] = file_sh[d];
    }

    if (need_slice) {
      // Calculate the cropping window, based on the final layout (user provides axes in that
      // layout)
      auto tmp_roi = slice_attr_.GetCropWindowGenerator(i)(sh.tensor_shape(i), {});
      sh.set_tensor_shape(i, tmp_roi.shape);  // set the final shape

      // Reverse the cropping window arguments if needed, as we do slice before transpose to save
      // bandwidth.
      auto &roi = rois_[i];
      if (is_transposed) {
        for (int d = 0; d < ndim; d++) {
          roi.anchor[d] = tmp_roi.anchor[ndim - 1 - d];
          roi.shape[d] = tmp_roi.shape[ndim - 1 - d];
        }
      } else {
        roi = std::move(tmp_roi);
      }
    }
  }
  output_desc.resize(1);
  output_desc[0].shape = std::move(sh);
  output_desc[0].type = output_type;
  return true;
}

void NumpyReader::RunImpl(HostWorkspace &ws) {
  auto &output = ws.OutputRef<CPUBackend>(0);
  const auto &out_sh = output.shape();
  int ndim = out_sh.sample_dim();
  int nsamples = out_sh.num_samples();
  auto &thread_pool = ws.GetThreadPool();
  int nthreads = thread_pool.NumThreads();
  scratch_.SetSize(nthreads);

  for (int i = 0; i < nsamples; i++) {
    const auto& file_i = GetSample(i);
    const auto& file_sh = file_i.image.shape();
    bool need_transpose = !(file_i.meta == "transpose:false");
    bool need_slice = !rois_.empty();

    // controls task priority
    int64_t task_sz = volume(file_i.image.shape());
    if (need_slice)
      task_sz += volume(rois_[i].shape);
    if (need_transpose)
      task_sz += out_sh.tensor_size(i);

    thread_pool.AddWork([&, i, need_transpose, need_slice](int tid) {
      if (need_slice && need_transpose) {
        scratch_[tid].Resize(rois_[i].shape);
        scratch_[tid].set_type(file_i.image.type());
        SliceHelper(scratch_[tid], file_i.image, rois_[i]);
        TransposeHelper(output[i], scratch_[tid]);
      } else if (need_slice) {
        SliceHelper(output[i], file_i.image, rois_[i]);
      } else if (need_transpose) {
        TransposeHelper(output[i], file_i.image);
      } else {
        std::memcpy(output[i].raw_mutable_data(), file_i.image.raw_data(), file_i.image.nbytes());
      }
      output[i].SetSourceInfo(file_i.image.GetSourceInfo());
    }, task_sz);
  }
  thread_pool.RunAll();
}

}  // namespace dali
