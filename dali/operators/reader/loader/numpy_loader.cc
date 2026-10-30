// Copyright (c) 2020-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "dali/operators/reader/loader/numpy_loader.h"
#include <dirent.h>
#include <errno.h>
#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>
#include "dali/core/common.h"
#include "dali/operators/reader/loader/filesystem.h"
#include "dali/operators/reader/loader/utils.h"
#include "dali/util/file.h"
#include "dali/util/uri.h"

namespace dali {
namespace detail {

bool NumpyHeaderCache::GetFromCache(const string &file_name, numpy::HeaderData &header) {
  if (!cache_headers_) {
    return false;
  }
  std::unique_lock<std::mutex> cache_lock(cache_mutex_);
  auto it = header_cache_.find(file_name);
  if (it == header_cache_.end()) {
    return false;
  } else {
    header = it->second;
    return true;
  }
}

void NumpyHeaderCache::UpdateCache(const string &file_name, const numpy::HeaderData &value) {
  if (cache_headers_) {
    std::unique_lock<std::mutex> cache_lock(cache_mutex_);
    header_cache_[file_name] = value;
  }
}

}  // namespace detail

void NumpyLoader::ReadSample(NumpyFileWrapper& target) {
  const auto& entry = file_entries_[current_index_++];
  auto filename = entry.filename;
  auto size = entry.size;

  // handle wrap-around
  MoveToNextShard(current_index_);

  // metadata info
  DALIMeta meta;
  meta.SetSourceInfo(filename);
  meta.SetSkipSample(false);

  // if data is cached, skip loading
  if (ShouldSkipImage(filename)) {
    meta.SetSkipSample(true);
    target.data.Reset();
    target.data.SetMeta(meta);
    target.data.Resize({0}, DALI_UINT8);
    target.filename.clear();
    return;
  }

  auto path = filesystem::join_path(file_root_, filename);
  FileStream::Options opts;
  opts.read_ahead = read_ahead_;
  opts.use_mmap = !copy_read_data_;
  opts.use_odirect = use_o_direct_;
  auto current_file = FileStream::Open(path, opts, size);

  // read the header
  numpy::HeaderData header;
  auto ret = header_cache_.GetFromCache(filename, header);
  size_t payload_bytes = 0;
  try {
    if (!ret) {
      if (opts.use_odirect) {
        numpy::ParseODirectHeader(header, current_file.get(), o_direct_alignm_,
                                  o_direct_read_len_alignm_);
      } else {
        numpy::ParseHeader(header, current_file.get());
      }
      header_cache_.UpdateCache(filename, header);
    }
    payload_bytes = header.nbytes();
  } catch (const std::runtime_error &e) {
    DALI_FAIL(e.what(), ". File: ", filename);
  }
  auto file_size = current_file->Size();
  auto max_payload = static_cast<size_t>(std::numeric_limits<Index>::max());
  DALI_ENFORCE(payload_bytes <= max_payload, make_string(
        "Numpy array payload exceeds the supported size: ", filename, "; requested ",
        payload_bytes, " bytes, maximum ", max_payload, " bytes."));
  DALI_ENFORCE(header.data_offset >= 0 && static_cast<uint64_t>(header.data_offset) <= file_size,
      make_string(
        "Numpy array data offset is outside the file: ", filename, "; offset ",
        header.data_offset, " bytes, file size ", file_size, " bytes."));
  auto available_bytes = file_size - header.data_offset;
  DALI_ENFORCE(payload_bytes <= available_bytes, make_string(
        "Numpy array payload exceeds the available file data: ", filename, "; requested ",
        payload_bytes, " bytes, available ", available_bytes, " bytes."));
  current_file->SeekRead(header.data_offset);
  Index nbytes = payload_bytes;
  target.shape = header.shape;
  target.type = header.type();
  target.meta = meta;
  target.data_offset = header.data_offset;
  target.nbytes = nbytes;
  target.filename = std::move(path);

  if (!opts.use_mmap || !current_file->CanMemoryMap()) {
    target.current_file = std::move(current_file);
  } else {
    auto p = current_file->Get(nbytes);
    DALI_ENFORCE(p != nullptr, make_string("Failed to read file: ", filename));
    // Wrap the raw data in the Tensor object.
    target.data.ShareData(p, nbytes, false, {nbytes}, header.type(), CPU_ONLY_DEVICE_ID);
    target.data.Resize(header.shape, header.type());
    // close the file handle
    current_file->Close();
  }
  target.data.SetMeta(meta);

  // set meta
  target.fortran_order = header.fortran_order;
}

void NumpyLoader::Skip() {
  MoveToNextShard(++current_index_);
}

}  // namespace dali
