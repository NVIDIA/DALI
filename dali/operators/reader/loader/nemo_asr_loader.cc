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
#include <glob.h>
#include <memory>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/nemo_asr_loader.h"
#include "dali/util/file.h"
#include "dali/operators/reader/loader/utils.h"
#include "dali/operators/decoder/audio/generic_decoder.h"

namespace dali {

namespace detail {

void ParseManifestFile(std::vector<NemoAsrEntry> &entries, const std::string &manifest_filepath) {
  entries.push_back({"/path/to/file1.wav", 3.1, 0.1, "this is a transcript 1"});
  entries.push_back({"/path/to/file2.wav", 3.2, 0.2, "this is a transcript 2"});
  entries.push_back({"/path/to/file3.wav", 3.3, 0.3, "this is a transcript 3"});
}

}  // namespace detail

void NemoAsrLoader::PrepareMetadataImpl() {
  detail::ParseManifestFile(entries_, manifest_filepath_);

  DALI_ENFORCE(Size() > 0, "No files found.");
  if (shuffle_) {
    // seeded with hardcoded value to get
    // the same sequence on every shard
    std::mt19937 g(kDaliDataloaderSeed);
    std::shuffle(entries_.begin(), entries_.end(), g);
  }
  Reset(true);
}

void NemoAsrLoader::Reset(bool wrap_to_shard) {
  current_index_ = wrap_to_shard ? start_index(shard_id_, num_shards_, Size()) : 0;
  current_epoch_++;

  if (shuffle_after_epoch_) {
    std::mt19937 g(kDaliDataloaderSeed + current_epoch_);
    std::shuffle(entries_.begin(), entries_.end(), g);
  }
}

void NemoAsrLoader::PrepareEmpty(AsrSample &asr) {
  PrepareEmptyTensor(asr);
}

void NemoAsrLoader::ReadSample(ImageFileWrapper& imfile) {
  auto &entry = entries_[current_index_++];

  // handle wrap-around
  MoveToNextShard(current_index_);

  // metadata info
  DALIMeta meta;
  meta.SetSourceInfo(entry.audio_filepath);
  meta.SetSkipSample(false);

  auto current_image = FileStream::Open(file_root_ + "/" + image_file, read_ahead_,
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
  imfile.filename = file_root_ + "/" + image_file;
}
*/
Index NemoAsrLoader::SizeImpl() {
  return entries_.size();
}
}  // namespace dali
