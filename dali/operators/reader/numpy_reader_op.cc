// Copyright (c) 2020-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_cpu.h"
#include "dali/kernels/transpose/transpose.h"
#include "dali/core/static_switch.h"
#include "dali/operators/reader/numpy_reader_op.h"

namespace dali {

static void CopyHelper(Tensor<CPUBackend> &output, const Tensor<CPUBackend> &input,
                       ThreadPool &thread_pool, int min_blk_sz, int req_nblocks) {
  auto out_ptr = static_cast<uint8_t*>(output.raw_mutable_data());
  auto in_ptr = static_cast<const uint8_t*>(input.raw_data());
  auto nelements = volume(input.shape());
  auto nbytes = input.nbytes();
  if (nelements <= min_blk_sz) {
    thread_pool.AddWork([=](int tid) {
      std::memcpy(out_ptr, in_ptr, nbytes);
    }, nelements);
  } else {
    int64_t prev_b_start = 0;
    auto task_sz = nelements / req_nblocks;
    for (int b = 0; b < req_nblocks; b++) {
      int64_t b_start = prev_b_start;
      int64_t b_end = prev_b_start = nbytes * (b + 1) / req_nblocks;
      int64_t b_size = b_end - b_start;
      thread_pool.AddWork([=](int tid) {
        std::memcpy(out_ptr + b_start, in_ptr + b_start, b_size);
      }, b_size);
    }
  }
}

static void TransposeHelper(Tensor<CPUBackend> &output, const Tensor<CPUBackend> &input) {
  int n_dims = input.shape().sample_dim();
  SmallVector<int, 6> perm;
  perm.resize(n_dims);
  for (int i = 0; i < n_dims; ++i)
    perm[i] = n_dims - i - 1;
  TYPE_SWITCH(input.type().id(), type2id, T, NUMPY_ALLOWED_TYPES, (
    kernels::TransposeGrouped(view<T>(output), view<const T>(input), make_cspan(perm));
  ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())));  // NOLINT
}

static void SliceHelper(Tensor<CPUBackend> &output, const Tensor<CPUBackend> &input,
                        const CropWindow &roi, float fill_value, ThreadPool &thread_pool,
                        int min_blk_sz, int req_nblocks) {
  int ndim = input.shape().sample_dim();
  VALUE_SWITCH(ndim, Dims, (1, 2, 3, 4, 5, 6), (
    TYPE_SWITCH(input.type().id(), type2id, T, NUMPY_ALLOWED_TYPES, (
      kernels::SliceCPU<T, T, Dims> kernel;
      kernels::SliceArgs<T, Dims> args;
      args.anchor = roi.anchor;
      args.shape = roi.shape;
      args.fill_values.clear();
      args.fill_values.push_back(ConvertSat<T>(fill_value));
      kernels::KernelContext ctx;
      auto out_view = view<T, Dims>(output);
      auto in_view = view<const T, Dims>(input);
      // no need to run Setup (we already know the output shape)
      kernel.Schedule(ctx, out_view, in_view, args, thread_pool, min_blk_sz, req_nblocks);
    ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported number of dimensions: ", ndim)););  // NOLINT
}

static void SlicePermuteHelper(Tensor<CPUBackend> &output, const Tensor<CPUBackend> &input,
                               const CropWindow &roi, float fill_value, ThreadPool &thread_pool,
                               int min_blk_sz, int req_nblocks) {
  const auto &in_shape = input.shape();
  int ndim = in_shape.sample_dim();
  VALUE_SWITCH(ndim, Dims, (1, 2, 3, 4, 5, 6), (
    TYPE_SWITCH(input.type().id(), type2id, T, NUMPY_ALLOWED_TYPES, (
      kernels::SliceFlipNormalizePermutePadCpu<T, T, Dims> kernel;
      kernels::SliceFlipNormalizePermutePadArgs<Dims> args(roi.shape, in_shape);
      args.anchor = roi.anchor;
      for (int d = 0; d < Dims; d++)
        args.permuted_dims[d] = Dims - 1 - d;
      args.fill_values.clear();
      args.fill_values.push_back(ConvertSat<T>(fill_value));
      kernels::KernelContext ctx;
      auto out_view = view<T, Dims>(output);
      auto in_view = view<const T, Dims>(input);
      // no need to run Setup (we already know the output shape)
      kernel.Schedule(ctx, out_view, in_view, args, thread_pool, min_blk_sz, req_nblocks);
    ), DALI_FAIL(make_string("Unsupported input type: ", input.type().id())));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported number of dimensions: ", ndim)););  // NOLINT
}

DALI_REGISTER_OPERATOR(readers__Numpy, NumpyReaderCPU, CPU);

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

.. warning::
    This argument is temporarily disabled and left for backward compatibility.
    It will be reenabled in the future releases.

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
    .AddOptionalArg("out_of_bounds_policy",
        R"code(Determines the policy when reading outside of the bounds of the numpy array.

Here is a list of the supported values:

- ``"error"`` (default): Attempting to read outside of the bounds of the image will produce an error.
- ``"pad"``: The array will be padded as needed with zeros or any other value that is specified
  with the ``fill_value`` argument.
- ``"trim_to_shape"``: The ROI will be cut to the bounds of the array.)code",
        "error")
    .AddOptionalArg("fill_value",
        R"code(Determines the padding value when ``out_of_bounds_policy`` is set to “pad”.)code",
        0.f)
  .AddParent("LoaderBase");


// Deprecated alias
DALI_REGISTER_OPERATOR(NumpyReader, NumpyReaderCPU, CPU);

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

void NumpyReaderCPU::RunImpl(HostWorkspace &ws) {
  auto &output = ws.OutputRef<CPUBackend>(0);
  const auto &out_sh = output.shape();
  int ndim = out_sh.sample_dim();
  int nsamples = out_sh.num_samples();
  auto &thread_pool = ws.GetThreadPool();
  int nthreads = thread_pool.NumThreads();

  // From 1 to 10 blocks per sample depending on the nthreads/nsamples ratio
  int blocks_per_sample = std::max(1, 10 * nthreads / nsamples);
  constexpr int kThreshold = kernels::kSliceMinBlockSize;  // smaller samples will not be subdivided

  for (int i = 0; i < nsamples; i++) {
    const auto& file_i = GetSample(i);
    const auto& file_sh = file_i.get_shape();
    int64_t sample_sz = volume(file_i.get_shape());
    if (need_slice_[i] && need_transpose_[i]) {
      SlicePermuteHelper(output[i], file_i.data, rois_[i], fill_value_, thread_pool, kThreshold,
                         blocks_per_sample);
    } else if (need_slice_[i]) {
      SliceHelper(output[i], file_i.data, rois_[i], fill_value_, thread_pool, kThreshold,
                  blocks_per_sample);
    } else if (need_transpose_[i]) {
      // TODO(janton): Parallelize when Transpose supports tiling
      thread_pool.AddWork([&, i](int tid) {
        TransposeHelper(output[i], file_i.data);
      }, sample_sz * 8);  // 8 x (heuristic)
    } else {
      CopyHelper(output[i], file_i.data, thread_pool, kThreshold, blocks_per_sample);
    }
  }
  thread_pool.RunAll();
}

}  // namespace dali
