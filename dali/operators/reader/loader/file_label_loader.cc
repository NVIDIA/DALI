// Copyright (c) 2017-2019, NVIDIA CORPORATION. All rights reserved.
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
#include "dali/operators/reader/loader/file_label_loader.h"
#include "dali/util/file.h"
#include "dali/operators/reader/loader/utils.h"

namespace dali {

using filesystem::dir_sep;

void FileLabelLoader::PrepareEmpty(ImageLabelWrapper &image_label) {
  PrepareEmptyTensor(image_label.image);
}

void FileLabelLoader::ReadSample(ImageLabelWrapper &image_label) {
  auto image_pair = image_label_pairs_[current_index_++];

  // handle wrap-around
  MoveToNextShard(current_index_);

  // copy the label
  image_label.label = image_pair.second;
  DALIMeta meta;
  meta.SetSourceInfo(image_pair.first);
  meta.SetSkipSample(false);

  // if image is cached, skip loading
  if (ShouldSkipImage(image_pair.first)) {
    meta.SetSkipSample(true);
    image_label.image.Reset();
    image_label.image.SetMeta(meta);
    image_label.image.set_type(TypeInfo::Create<uint8_t>());
    image_label.image.Resize({0});
    return;
  }

  auto current_image = FileStream::Open(filesystem::join_path(file_root_, image_pair.first),
                                        read_ahead_, !copy_read_data_);
  Index image_size = current_image->Size();

  if (copy_read_data_) {
    if (image_label.image.shares_data()) {
      image_label.image.Reset();
    }
    image_label.image.Resize({image_size});
    // copy the image
    Index ret = current_image->Read(image_label.image.mutable_data<uint8_t>(), image_size);
    DALI_ENFORCE(ret == image_size, make_string("Failed to read file: ", image_pair.first));
  } else {
    auto p = current_image->Get(image_size);
    DALI_ENFORCE(p != nullptr, make_string("Failed to read file: ", image_pair.first));
    // Wrap the raw data in the Tensor object.
    image_label.image.ShareData(p, image_size, {image_size});
    image_label.image.set_type(TypeInfo::Create<uint8_t>());
  }

  // close the file handle
  current_image->Close();

  image_label.image.SetMeta(meta);
}

Index FileLabelLoader::SizeImpl() {
  return static_cast<Index>(image_label_pairs_.size());
}
}  // namespace dali
