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

#include <dirent.h>
#include <errno.h>
#include <memory>
#include <set>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/numpy_loader_gpu.h"

namespace dali {

// register tensor
void NumpyLoaderGPU::RegisterTensor(void *buffer, size_t total_size) {
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
        cuFileBufDeregister(iter->first);
      }
    }
    // no buffer found or with different size so register again
    cuFileBufRegister(dptr, total_size, 0);
    reg_lock.lock();
    reg_buff_[dptr] = total_size;
  }
}

void NumpyLoaderGPU::ReadSampleHelper(CUFileStream *file, ImageFileWrapperGPU& imfile,
                                      void *buffer, Index offset, size_t total_size) {
  // register the buffer (if needed)
  RegisterTensor(buffer, total_size);

  Index image_bytes = volume(imfile.shape) * imfile.type_info.size();

  // copy the image
  file->Read(static_cast<uint8_t*>(buffer), image_bytes, offset);
}

// we need to implement that but we should split parsing and reading in this case
void NumpyLoaderGPU::ReadSample(ImageFileWrapperGPU& imfile) {
  // set the device:
  DeviceGuard g(device_id_);

  // extract image file
  auto image_file = images_[current_index_++];

  // handle wrap-around
  MoveToNextShard(current_index_);

  // metadata info
  DALIMeta meta;
  meta.SetSourceInfo(image_file);
  meta.SetSkipSample(false);

  // if image is cached, skip loading
  if (ShouldSkipImage(image_file)) {
    meta.SetSkipSample(true);
    imfile.image.SetMeta(meta);
    imfile.image.Reset();
    imfile.image.set_type(TypeInfo::Create<uint8_t>());
    imfile.image.Resize({0});
    imfile.filename.clear();
    return;
  }

  // set metadata
  imfile.image.SetMeta(meta);

  imfile.read_meta_f = [this, image_file, &imfile] () {
    // open file
    imfile.file_stream = CUFileStream::Open(file_root_ + "/" + image_file, read_ahead_, false);

    // read the header
    NumpyParseTarget target;
    auto ret = header_cache_.GetFromCache(image_file, target);
    if (ret) {
      imfile.file_stream->Seek(target.data_offset);
    } else {
      detail::ParseHeader(imfile.file_stream.get(), target, &CUFileStream::ReadCPU);
      header_cache_.UpdateCache(image_file, target);
    }

    imfile.type_info = target.type_info;
    imfile.shape = target.shape;
    imfile.meta = (target.fortran_order ? "transpose:true" : "transpose:false");
  };

  imfile.read_sample_f = [this, image_file, &imfile] (void *buffer, Index offset,
                                                      size_t total_size) {
    // read sample
    ReadSampleHelper(imfile.file_stream.get(), imfile, buffer, offset, total_size);
    // close the file handle
    imfile.file_stream->Close();
  };

  // set file path
  imfile.filename = file_root_ + "/" + image_file;
}

}  // namespace dali
