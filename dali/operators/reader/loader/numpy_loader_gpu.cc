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

#include <dirent.h>
#include <errno.h>
#include <memory>
#include <set>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/numpy_loader_gpu.h"

namespace dali {

// register buffer
void NumpyLoaderGPU::RegisterBuffer(void *buffer, size_t total_size) {
  if (register_buffers_) {
    // get raw pointer
    auto dptr = static_cast<uint8_t*>(buffer);
    std::unique_lock<std::mutex> reg_lock(reg_mutex_);
    auto iter = reg_buff_.find(dptr);
    reg_lock.unlock();
    if (iter != reg_buff_.end()) {
      if (iter->second == total_size) {
        // the right buffer is registered already, return
        return;
      } else {
        CUDA_CALL(cuFileBufDeregister(iter->first));
      }
    }
    // no buffer found or with different size so register again
    CUDA_CALL(cuFileBufRegister(dptr, total_size, 0));
    reg_lock.lock();
    reg_buff_[dptr] = total_size;
  }
}

void NumpyLoaderGPU::ReadSampleHelper(CUFileStream *file,
                                      void *buffer, Index file_offset, size_t read_size) {
  // register the buffer (if needed)
  RegisterBuffer(buffer, read_size);

  // copy the image
  file->ReadGPUImpl(static_cast<uint8_t*>(buffer), read_size, 0, file_offset);
}

void NumpyLoaderGPU::PrepareEmpty(NumpyFileWrapperGPU& target) {
  target = {};
}

// we need to implement that but we should split parsing and reading in this case
void NumpyLoaderGPU::ReadSample(NumpyFileWrapperGPU& target) {
  // set the device:
  DeviceGuard g(device_id_);

  // extract image file
  auto filename = files_[current_index_++];

  // handle wrap-around
  MoveToNextShard(current_index_);

  // metadata info
  DALIMeta meta;
  meta.SetSourceInfo(filename);
  meta.SetSkipSample(false);

  // if image is cached, skip loading
  if (ShouldSkipImage(filename)) {
    meta.SetSkipSample(true);
    target.meta = meta;
    target.filename.clear();
    return;
  }

  // set metadata
  target.meta = meta;

  target.read_meta_f = [this, filename, &target] () {
    // open file
    target.file_stream = CUFileStream::Open(file_root_ + "/" + filename, read_ahead_, false);

    // read the header
    NumpyParseTarget parse_target;
    auto ret = header_cache_.GetFromCache(filename, parse_target);
    if (ret) {
      target.file_stream->Seek(parse_target.data_offset);
    } else {
      detail::ParseHeader(target.file_stream.get(), parse_target);
      header_cache_.UpdateCache(filename, parse_target);
    }

    target.type = parse_target.type_info;
    target.shape = parse_target.shape;
    target.fortran_order = parse_target.fortran_order;
  };

  target.read_sample_f = [this, filename, &target] (void *buffer, Index file_offset,
                                                      size_t read_size) {
    // read sample
    ReadSampleHelper(target.file_stream.get(), buffer, file_offset, read_size);
    // we cannot close the file handle here, we need to remember to do it later on
  };

  // set file path
  target.filename = file_root_ + "/" + filename;
}

}  // namespace dali
