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

#include "dali/operators/reader/loader/numpy_loader_gpu.h"
#include <dirent.h>
#include <errno.h>
#include <memory>
#include <set>
#include "dali/core/common.h"
#include "dali/operators/reader/loader/filesystem.h"

namespace dali {

void NumpyLoaderGPU::PrepareEmpty(NumpyFileWrapperGPU& target) {
  target = {};
}

void NumpyFileWrapperGPU::Reopen() {
  FileStream::Options opts;
  opts.read_ahead = read_ahead;
  opts.use_mmap = false;
  opts.use_odirect = false;
  file_stream_ = CUFileStream::Open(filename, opts);
}

void NumpyFileWrapperGPU::ReadHeader(detail::NumpyHeaderCache &cache) {
  numpy::HeaderData header;
  bool ret = cache.GetFromCache(filename, header);
  try {
    if (!ret) {
      numpy::ParseHeader(header, file_stream_.get());
      cache.UpdateCache(filename, header);
    }
  } catch (const std::runtime_error &e) {
    DALI_FAIL(e.what(), ". File: ", filename);
  }
  file_stream_->SeekRead(header.data_offset);

  type = header.type();
  shape = header.shape;
  fortran_order = header.fortran_order;
  data_offset = header.data_offset;
}

void NumpyFileWrapperGPU::ReadRawChunk(void* buffer, size_t bytes,
                                       Index buffer_offset, Index file_offset) {
  file_stream_->ReadAtGPU(static_cast<uint8_t *>(buffer),
                         bytes, buffer_offset, file_offset);
}

// we need to implement that but we should split parsing and reading in this case
void NumpyLoaderGPU::ReadSample(NumpyFileWrapperGPU& target) {
  // set the device:
  DeviceGuard g(device_id_);

  // extract image file
  auto filename = file_entries_[current_index_++].filename;

  // handle wrap-around
  MoveToNextShard(current_index_);

  // metadata info
  DALIMeta meta;
  meta.SetSourceInfo(filename);
  meta.SetSkipSample(false);

  // set file path
  target.filename = filesystem::join_path(file_root_, filename);

  // if image is cached, skip loading
  if (ShouldSkipImage(filename)) {
    meta.SetSkipSample(true);
    target.meta = meta;
    target.filename.clear();
    return;
  }

  // set metadata
  target.meta = meta;
  target.read_ahead = read_ahead_;
}

}  // namespace dali
