// Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <memory>

#include "dali/core/backend_tags.h"
#include "dali/kernels/slice/slice_cpu.h"
#include "dali/kernels/slice/slice_flip_normalize_permute_pad_cpu.h"
#include "dali/kernels/transpose/transpose.h"
#include "dali/core/static_switch.h"
#include "dali/operators/reader/numpy_reader_op.h"
#include "dali/pipeline/data/backend.h"
#include "dali/core/mm/memory.h"

namespace dali {

static void CopyHelper(SampleView<CPUBackend> output, ConstSampleView<CPUBackend> input,
                       ThreadPool &thread_pool, int min_blk_sz, int req_nblocks) {
  auto *out_ptr = static_cast<uint8_t *>(output.raw_mutable_data());
  const auto *in_ptr = static_cast<const uint8_t *>(input.raw_data());
  auto nelements = input.shape().num_elements();
  auto nbytes = nelements * TypeTable::GetTypeInfo(input.type()).size();
  if (nelements <= min_blk_sz) {
    thread_pool.AddWork([=](int tid) {
      std::memcpy(out_ptr, in_ptr, nbytes);
    }, nelements);
  } else {
    int64_t prev_b_start = 0;
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

static void SliceHelper(SampleView<CPUBackend> output, ConstSampleView<CPUBackend> input,
                        const CropWindow &roi, float fill_value, ThreadPool &thread_pool,
                        int min_blk_sz, int req_nblocks) {
  int ndim = input.shape().sample_dim();
  VALUE_SWITCH(ndim, Dims, (1, 2, 3, 4, 5, 6), (
    TYPE_SWITCH(input.type(), type2id, T, NUMPY_ALLOWED_TYPES, (
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
    ), DALI_FAIL(make_string("Unsupported input type: ", input.type())));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported number of dimensions: ", ndim)););  // NOLINT
}

static void SlicePermuteHelper(SampleView<CPUBackend> output, ConstSampleView<CPUBackend> input,
                               const CropWindow &roi, float fill_value, ThreadPool &thread_pool,
                               int min_blk_sz, int req_nblocks) {
  const auto &in_shape = input.shape();
  int ndim = in_shape.sample_dim();
  VALUE_SWITCH(ndim, Dims, (1, 2, 3, 4, 5, 6), (
    TYPE_SWITCH(input.type(), type2id, T, NUMPY_ALLOWED_TYPES, (
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
    ), DALI_FAIL(make_string("Unsupported input type: ", input.type())));  // NOLINT
  ), DALI_FAIL(make_string("Unsupported number of dimensions: ", ndim)););  // NOLINT
}

DALI_REGISTER_OPERATOR(readers__Numpy, NumpyReaderCPU, CPU);

DALI_SCHEMA(readers__Numpy)
  .DocStr(R"(Reads Numpy arrays from a directory.

This operator can be used in the following modes:

1. Read all files from a directory indicated by `file_root` that match given `file_filter`.
2. Read file names from a text file indicated in `file_list` argument.
3. Read files listed in `files` argument.

.. note::
  The ``gpu`` backend requires cuFile/GDS support (418.x driver family or newer). which is
  shipped with the CUDA toolkit starting from CUDA 11.4. Please check the GDS documentation
  for more details.

  The ``gpu`` reader reads the files in chunks. The size of the chunk can be controlled
  process-wide with an environment variable ``DALI_GDS_CHUNK_SIZE``. Valid values are powers of 2
  between 4096 and 16M, with the default being 2M. For convenience, the value can be specified
  with a k or M suffix, applying a multiplier of 1024 and 2^20, respectively.
)")
  .NumInput(0)
  .NumOutput(1)  // (Arrays)
  .AddOptionalArg<string>("file_root",
      R"(Path to a directory that contains the data files.

If not using `file_list` or `files`. this directory is traversed to discover the files.
`file_root` is required in this mode of operation.)",
      nullptr)
  .AddOptionalArg("file_filter",
      R"(If a value is specified, the string is interpreted as glob string to filter the
list of files in the sub-directories of the `file_root`.

This argument is ignored when file paths are taken from `file_list` or `files`.)", "*.npy")
  .AddOptionalArg<string>("file_list",
      R"(Path to a text file that contains filenames (one per line)
where the filenames are relative to the location of that file or to `file_root`, if specified.

This argument is mutually exclusive with `files`.)", nullptr)
.AddOptionalArg("shuffle_after_epoch",
      R"(If set to True, the reader shuffles the entire dataset after each epoch.

`stick_to_shard` and `random_shuffle` cannot be used when this argument is set to True.)",
      false)
  .AddOptionalArg<vector<string>>("files", R"(A list of file paths to read the data from.

If `file_root` is provided, the paths are treated as being relative to it.

This argument is mutually exclusive with `file_list`.)", nullptr)
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
        R"code(Order of dimensions used for the ROI anchor and shape arguments, as dimension indices.

If not provided, all the dimensions should be specified in the ROI arguments.
)code",
        std::vector<int>{})
    .AddOptionalArg("out_of_bounds_policy",
        R"code(Determines the policy when reading outside of the bounds of the numpy array.

Here is a list of the supported values:

- ``"error"`` (default): Attempting to read outside of the bounds of the image will produce an error.
- ``"pad"``: The array will be padded as needed with zeros or any other value that is specified
  with the `fill_value` argument.
- ``"trim_to_shape"``: The ROI will be cut to the bounds of the array.)code",
        "error")
    .AddOptionalArg("fill_value",
        R"code(Determines the padding value when `out_of_bounds_policy` is set to “pad”.)code",
        0.f)
    .AddOptionalArg("use_o_direct",
      R"code(If set to True, the data will be read directly from the storage bypassing system
cache.

Mutually exclusive with ``dont_use_mmap=False``.)code",
      false)
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
        "1.0",
        "readers__Numpy",
        R"code(In DALI 1.0 all readers were moved into a dedicated :mod:`~nvidia.dali.fn.readers`
submodule and renamed to follow a common pattern. This is a placeholder operator with identical
functionality to allow for backward compatibility.)code");

NumpyReaderCPU::~NumpyReaderCPU() {
  // Stop the prefetch thread as it uses the thread pool from this class. So before we can
  // destroy the thread pool make sure no one is using it anymore.
  this->StopPrefetchThread();
}

void NumpyReaderCPU::Prefetch() {
  // We actually prepare the next batch
  DomainTimeRange tr("[DALI][NumpyReaderCPU] Prefetch #" + to_string(curr_batch_producer_),
                      DomainTimeRange::kRed);
  DataReader<CPUBackend, NumpyFileWrapper, NumpyFileWrapper, true>::Prefetch();

  if (!dont_use_mmap_)
    return;
  auto &curr_batch = prefetched_batch_queue_[curr_batch_producer_];

  string previous_path;
  for (unsigned idx = 0; idx < curr_batch.size(); ++idx) {
    // in case of pad_last_batch the curr_batch elements are pointing to the same object
    // including the data, so there it no need to read it again or it can even lead to a race
    // with allocation/deallocation of memory and concurrent read
    if (idx > 0 && curr_batch[idx - 1] == curr_batch[idx]) {
      break;
    }
    auto &target = curr_batch[idx];

    // if we pad last batch but we duplicate the samples from the previous one - a case
    // with multiple unequal shards where we need to create a full duplicated batch
    // so there is no need to read again this data
    if (!target->current_file) {
      break;
    }
    if (target->data.shares_data()) {
      target->data.Reset();
    }
    if (use_o_direct_) {
      /*
       * the offset to the data we are interested in
       *                  |<-                       aligned_len                    ->|
       *                  |                                                          |
       *                  |                               |<- target->nbytes ->|     |
       *                  |<-    target_data_offset     ->|                    |     |
       *     |<-          target->data_offset           ->|
       *     -------------********************************XXXXXXXXXXXXXXXXXXXXXX******
       *     |            |                               |                          |
       * file_start   block_start/allocated_memory     shared_ptr                   block_end
       */

      // align the size of data to read accounting for its start
      auto block_start = align_down(target->data_offset, o_direct_alignm_);
      auto block_end = align_up(target->data_offset + target->nbytes, o_direct_alignm_);
      auto aligned_len = align_up(block_end - block_start, o_direct_read_len_alignm_);
      // offset to the desired data from the block start
      auto target_data_offset = alignment_offset(target->data_offset, o_direct_alignm_);
      // allocate the memory that is aligned to the block size, but wrap it into shared ptr using
      auto resource_mem = mm::alloc_raw_shared<char, mm::memory_kind::host>(aligned_len,
                                                                            o_direct_alignm_);
      shared_ptr<void> tmp_mem(resource_mem, resource_mem.get() + target_data_offset);

      // split data into chunks and copy separately
      auto file = dynamic_cast<ODirectFileStream*>(target->current_file.get());
      auto read_tail = alignment_offset(target_data_offset + target->nbytes, o_direct_chunk_size_);
      for (size_t read_offset = 0; read_offset < aligned_len; read_offset += o_direct_chunk_size_) {
        // read whole chunk or just aligned number of blocks to match aligned_len
        auto read_size = std::min(o_direct_chunk_size_, aligned_len - read_offset);
        auto target_mem = static_cast<char*>(tmp_mem.get()) - target_data_offset + read_offset;
        // where to read from counting from the file start
        auto file_offset = read_offset + align_down(target->data_offset, o_direct_alignm_);
        thread_pool_.AddWork([this, &target, file, read_size, target_mem, file_offset, read_tail]
                             (int tid) {
          Index ret = file->ReadAt(target_mem, read_size, file_offset);
          DALI_ENFORCE(ret >= static_cast<Index>(read_tail) &&
                       ret <= static_cast<Index>(o_direct_chunk_size_),
                       make_string("Failed to read file: ", target->filename,
                                   ", read: ", ret, " while it should be [", read_tail, ", ",
                                   o_direct_chunk_size_, "]"));
        });
      }
      target->data.ShareData(tmp_mem, target->nbytes, false, target->shape, target->type, -1);
    } else {
      if (!target->data.has_data()) target->data.set_pinned(false);
      target->data.Resize(target->shape, target->type);
      auto data_ptr = static_cast<uint8_t*>(target->data.raw_mutable_data());
      Index ret = target->current_file->Read(data_ptr, target->nbytes);
      DALI_ENFORCE(ret == static_cast<Index>(target->nbytes),
                  make_string("Failed to read file: ", target->filename,
                              ", read: ", ret, " while it should be ", target->nbytes));
    }
  }
  thread_pool_.RunAll();
  for (auto &target : curr_batch) {
    target->current_file.reset();
  }
}

void NumpyReaderCPU::RunImpl(Workspace &ws) {
  auto &output = ws.Output<CPUBackend>(0);
  const auto &out_sh = output.shape();
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
    auto input_sample = const_sample_view(file_i.data);
    if (need_slice_[i] && need_transpose_[i]) {
      SlicePermuteHelper(output[i], input_sample, rois_[i], fill_value_, thread_pool, kThreshold,
                         blocks_per_sample);
    } else if (need_slice_[i]) {
      SliceHelper(output[i], input_sample, rois_[i], fill_value_, thread_pool, kThreshold,
                  blocks_per_sample);
    } else if (need_transpose_[i]) {
      // TODO(janton): Parallelize when Transpose supports tiling
      thread_pool.AddWork([&, i, input_sample](int tid) {
        numpy::FromFortranOrder(output[i], input_sample);
      }, sample_sz * 8);  // 8 x (heuristic)
    } else {
      CopyHelper(output[i], input_sample, thread_pool, kThreshold, blocks_per_sample);
    }
    output.SetMeta(i, file_i.get_meta());
  }
  thread_pool.RunAll();
}

}  // namespace dali
