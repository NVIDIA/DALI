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

#include <string>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/operators/decoder/audio/generic_decoder.h"
#include "dali/operators/reader/loader/nemo_asr_loader.h"
#include "dali/operators/reader/loader/utils.h"
#include "dali/pipeline/data/views.h"
#include "dali/pipeline/util/lookahead_parser.h"
#include "dali/util/file.h"

namespace dali {

namespace detail {

void ParseManifest(std::vector<NemoAsrEntry> &entries, const std::string &json) {
  detail::LookaheadParser parser(const_cast<char*>(json.c_str()));

  RAPIDJSON_ASSERT(parser.PeekType() == kArrayType);
  parser.EnterArray();

  while (parser.NextArrayValue()) {
    if (parser.PeekType() != kObjectType)
      continue;
    parser.EnterObject();
    NemoAsrEntry entry;
    while (const char* key = parser.NextObjectKey()) {
      if (0 == detail::safe_strcmp(key, "audio_filepath")) {
        entry.audio_filepath = parser.GetString();
      } else if (0 == detail::safe_strcmp(key, "duration")) {
        entry.duration = parser.GetDouble();
      } else if (0 == detail::safe_strcmp(key, "offset")) {
        entry.offset = parser.GetDouble();
      } else if (0 == detail::safe_strcmp(key, "text")) {
        entry.text = parser.GetString();
      } else {
        parser.SkipValue();
      }
    }
    entries.emplace_back(std::move(entry));
  }
}

}  // namespace detail

void NemoAsrLoader::PrepareMetadataImpl() {
  std::ifstream fstream(manifest_filepath_);
  DALI_ENFORCE(fstream, make_string("Could not open NEMO ASR manifest file: \"", manifest_filepath_, "\""));
  std::string json((std::istreambuf_iterator<char>(fstream)), std::istreambuf_iterator<char>());
  detail::ParseManifest(entries_, json);

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

void NemoAsrLoader::PrepareEmpty(AsrSample &sample) {
  PrepareEmptyTensor(sample.audio);
  PrepareEmptyTensor(sample.text);
}

void NemoAsrLoader::ReadSample(AsrSample& sample) {
  auto &entry = entries_[current_index_++];

  // handle wrap-around
  MoveToNextShard(current_index_);

  // metadata info
  DALIMeta meta;
  meta.SetSourceInfo(entry.audio_filepath);
  meta.SetSkipSample(false);
  sample.audio.SetMeta(meta);
  DALI_ENFORCE(copy_read_data_, "Sharing data is not supported with this loader");
  
  // TODO(janton): do not create a new decoder each time
  GenericAudioDecoder<float> decoder;
  sample.audio_meta = decoder.OpenFromFile(entry.audio_filepath);
  sample.audio.set_type(TypeTable::GetTypeInfo(DALI_FLOAT));
  sample.audio.Resize({sample.audio_meta.length});
  int64_t decoded_size = decoder.DecodeTyped({sample.audio.mutable_data<float>(), sample.audio.size()});
  DALI_ENFORCE(decoded_size == sample.audio_meta.length);
  decoder.Close();

  sample.text.set_type(TypeTable::GetTypeInfo(DALI_UINT8));
  sample.text.Resize({entry.text.length() + 1});
  std::memcpy(sample.text.raw_mutable_data(), entry.text.c_str(), entry.text.length());
  sample.text.raw_mutable_data()[entry.text.length()] = '\0';
}

Index NemoAsrLoader::SizeImpl() {
  return entries_.size();
}
}  // namespace dali
