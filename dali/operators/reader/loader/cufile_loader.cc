// Copyright (c) 2020, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <glob.h>
#include <memory>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/cufile_loader.h"
#include "dali/operators/reader/loader/file_loader.h"
#include "dali/operators/reader/loader/utils.h"
#include "dali/util/cufile.h"
#include "dali/util/cufile_helper.h"

namespace dali {

CUFileLoader::CUFileLoader(const OpSpec& spec, vector<std::string> images, bool shuffle_after_epoch)
    : FileLoader<GPUBackend, ImageFileWrapperGPU, CUFileStream>(spec) {
  // set the device first
  DeviceGuard g(device_id_);

  // this is needed for the driver singleton
  static std::mutex open_driver_mutex;
  static std::weak_ptr<cufile::CUFileDriverHandle> driver_handle;

  // load the cufile driver
  std::lock_guard<std::mutex> dlock(open_driver_mutex);
  if (!(d_ = driver_handle.lock())) {
    d_ = std::make_shared<cufile::CUFileDriverHandle>(device_id_);
    driver_handle = d_;
  }
}

void CUFileLoader::PrepareEmpty(ImageFileWrapperGPU& image_file) {
  PrepareEmptyTensor(image_file.image);
  image_file.filename.clear();
}

void CUFileLoader::ReadSample(ImageFileWrapperGPU& imfile) {
  // set the device first
  DeviceGuard g(device_id_);
  this->FileLoader<GPUBackend, ImageFileWrapperGPU, CUFileStream>::ReadSample(imfile);
}

}  // namespace dali
