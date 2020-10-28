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

#include <memory>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/file_loader.h"
#include "dali/util/file.h"
#include "dali/operators/reader/loader/utils.h"

namespace dali {

void FileLoader::PrepareEmpty(ImageFileWrapper &image_file) {
  PrepareEmptyTensor(image_file.image);
  image_file.filename = "";
}

void FileLoader::ReadSample(ImageFileWrapper& imfile) {
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
    imfile.filename = "";
    return;
  }

  auto current_image = FileStream::Open(filesystem::join_path(file_root_, image_file), read_ahead_,
                                        !copy_read_data_);
  Index image_size = current_image->Size();

  if (copy_read_data_) {
    if (imfile.image.shares_data()) {
      imfile.image.Reset();
    }
    imfile.image.Resize({image_size});
    // copy the image
    Index ret = current_image->Read(imfile.image.mutable_data<uint8_t>(), image_size);
    DALI_ENFORCE(ret == image_size, make_string("Failed to read file: ", image_file));
  } else {
    auto p = current_image->Get(image_size);
    DALI_ENFORCE(p != nullptr, make_string("Failed to read file: ", image_file));
    // Wrap the raw data in the Tensor object.
    imfile.image.ShareData(p, image_size, {image_size});
    imfile.image.set_type(TypeInfo::Create<uint8_t>());
  }

  // close the file handle
  current_image->Close();

  // set metadata
  imfile.image.SetMeta(meta);

  // set string
  imfile.filename = filesystem::join_path(file_root_, image_file);
}

Index FileLoader::SizeImpl() {
  return static_cast<Index>(images_.size());
}
}  // namespace dali
