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

#include <glob.h>
#include <dirent.h>
#include <errno.h>
#include <memory>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/file_loader.h"
#include "dali/operators/reader/loader/cufile_loader.h"
#include "dali/util/cufile_helper.h"
#include "dali/util/cufile.h"
#include "dali/operators/reader/loader/utils.h"

namespace dali {

// this is needed for the driver singleton
static std::mutex open_driver_mutex;
static std::weak_ptr<cufile::CUFileDriverHandle> driver_handle;

CUFileLoader::CUFileLoader(const OpSpec& spec, vector<std::string> images,
                           bool shuffle_after_epoch)
    : Loader<GPUBackend, ImageFileWrapperGPU >(spec),
      file_filter_(spec.GetArgument<string>("file_filter")),
      images_(std::move(images)),
      shuffle_after_epoch_(shuffle_after_epoch),
      current_index_(0),
      current_epoch_(0) {

    vector<string> files;

    has_files_arg_ = spec.TryGetRepeatedArgument(files, "files");
    has_file_list_arg_ = spec.TryGetArgument(file_list_, "file_list");
    has_file_root_arg_ = spec.TryGetArgument(file_root_, "file_root");

    DALI_ENFORCE(has_file_root_arg_ || has_files_arg_ || has_file_list_arg_,
      "``file_root`` argument is required when not using ``files`` or ``file_list``.");

    DALI_ENFORCE(has_files_arg_ + has_file_list_arg_ <= 1,
      "File paths can be provided through ``files`` or ``file_list`` but not both.");

    if (has_file_list_arg_) {
      DALI_ENFORCE(!file_list_.empty(), "``file_list`` argument cannot be empty");
      if (!has_file_root_arg_) {
        auto idx = file_list_.rfind(filesystem::dir_sep);
        if (idx != string::npos) {
          file_root_ = file_list_.substr(0, idx);
        }
      }
    }

    if (has_files_arg_) {
      DALI_ENFORCE(files.size() > 0, "``files`` specified an empty list.");
      images_ = std::move(files);
    }
  /*
   * Those options are mutually exclusive as `shuffle_after_epoch` will make every shard looks differently
   * after each epoch so coexistence with `stick_to_shard` doesn't make any sense
   * Still when `shuffle_after_epoch` we will set `stick_to_shard` internally in the FileLabelLoader so all
   * DALI instances will do shuffling after each epoch
   */
  if (shuffle_after_epoch_ || stick_to_shard_)
    DALI_ENFORCE(!(shuffle_after_epoch_  && stick_to_shard_),
                 "shuffle_after_epoch and stick_to_shard cannot be both true");
  if (shuffle_after_epoch_ || shuffle_)
    DALI_ENFORCE(!(shuffle_after_epoch_ && shuffle_),
                 "shuffle_after_epoch and random_shuffle cannot be both true");
  /*
   * Imply `stick_to_shard` from  `shuffle_after_epoch
   */
  if (shuffle_after_epoch_) {
    stick_to_shard_ = true;
  }

  // set the device first
  DeviceGuard g(device_id_);

  // load the cufile driver
  std::lock_guard<std::mutex> dlock(open_driver_mutex);
  if (!(d_ = driver_handle.lock())) {
    d_ = std::make_shared<cufile::CUFileDriverHandle>(device_id_);
    driver_handle = d_;
  }

  // we do not support mmap yet
  copy_read_data_ = true;
}

void CUFileLoader::PrepareEmpty(ImageFileWrapperGPU &image_file) {
  PrepareEmptyTensor(image_file.image);
  image_file.filename.clear();
}

void CUFileLoader::ReadSample(ImageFileWrapperGPU& imfile) {
  // set the device first
  DeviceGuard g(device_id_);

  // get image file name
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
    imfile.image.Reset();
    imfile.image.SetMeta(meta);
    imfile.image.set_type(TypeInfo::Create<uint8_t>());
    imfile.image.Resize({0});
    imfile.filename.clear();
    return;
  }

  imfile.read_meta_f = {};
  imfile.read_sample_f = {};

  auto current_image = CUFileStream::Open(file_root_ + "/" + image_file, read_ahead_, false);
  Index image_size = current_image->Size();

  // we always have to copy since GDS does not support mmap yet
  if (imfile.image.shares_data()) {
    imfile.image.Reset();
  }
  imfile.image.Resize({image_size});
  // copy the image
  current_image->Read(imfile.image.mutable_data<uint8_t>(), image_size);

  // close the file handle
  current_image->Close();

  // set metadata
  imfile.image.SetMeta(meta);

  // set string
  imfile.filename = file_root_ + "/" + image_file;
}

Index CUFileLoader::SizeImpl() {
  return static_cast<Index>(images_.size());
}
}  // namespace dali
