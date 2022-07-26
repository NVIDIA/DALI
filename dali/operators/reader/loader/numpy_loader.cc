// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <dirent.h>
#include <errno.h>
#include <cstdlib>
#include <memory>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/numpy_loader.h"
#include "dali/util/file.h"
#include "dali/operators/reader/loader/utils.h"

namespace dali {
namespace detail {

bool NumpyHeaderCache::GetFromCache(const string &file_name, numpy::HeaderMeta &header) {
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

void NumpyHeaderCache::UpdateCache(const string &file_name, const numpy::HeaderMeta &value) {
  if (cache_headers_) {
    std::unique_lock<std::mutex> cache_lock(cache_mutex_);
    header_cache_[file_name] = value;
  }
}

}  // namespace detail

void NumpyLoader::ReadSample(NumpyFileWrapper& target) {
  auto filename = files_[current_index_++];

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
  auto current_file = FileStream::Open(path, read_ahead_, !copy_read_data_);

  // read the header
  numpy::HeaderMeta header;
  auto ret = header_cache_.GetFromCache(filename, header);
  try {
    if (ret) {
      current_file->SeekRead(header.data_offset);
    } else {
      header = numpy::ParseHeader(current_file.get());
      header_cache_.UpdateCache(filename, header);
    }
  } catch (const std::runtime_error &e) {
    DALI_FAIL(e.what() + ". File: " + filename);
  }

  Index nbytes = header.nbytes();

  if (copy_read_data_) {
    if (target.data.shares_data()) {
      target.data.Reset();
    }
    target.data.Resize(header.shape, header.type());
    // copy the image
    Index ret = current_file->Read(static_cast<uint8_t*>(target.data.raw_mutable_data()),
                                    nbytes);
    DALI_ENFORCE(ret == nbytes, make_string("Failed to read file: ", filename));
  } else {
    auto p = current_file->Get(nbytes);
    DALI_ENFORCE(p != nullptr, make_string("Failed to read file: ", filename));
    // Wrap the raw data in the Tensor object.
    target.data.ShareData(p, nbytes, false, {nbytes}, header.type(), CPU_ONLY_DEVICE_ID);
    target.data.Resize(header.shape, header.type());
  }

  // close the file handle
  current_file->Close();

  // set metadata
  target.data.SetMeta(meta);

  // set file path
  target.filename = std::move(path);

  // set meta
  target.fortran_order = header.fortran_order;
}

}  // namespace dali
