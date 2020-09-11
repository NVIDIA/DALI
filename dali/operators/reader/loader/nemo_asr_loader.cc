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
#include "dali/operators/decoder/audio/audio_decoder_impl.h"
#include "dali/operators/reader/loader/nemo_asr_loader.h"
#include "dali/operators/reader/loader/utils.h"
#include "dali/pipeline/data/views.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/pipeline/util/lookahead_parser.h"
#include "dali/util/file.h"

namespace dali {

namespace detail {

void ParseManifest(std::vector<NemoAsrEntry> &entries, std::istream& manifest_file,
                   float min_duration, float max_duration) {
  std::string line;
  while (std::getline(manifest_file, line)) {
    detail::LookaheadParser parser(const_cast<char*>(line.c_str()));
    if (parser.PeekType() != kObjectType) {
      DALI_WARN(make_string("Skipping invalid manifest line: ", line));
      continue;
    }
    parser.EnterObject();
    NemoAsrEntry entry;
    while (const char* key = parser.NextObjectKey()) {
      if (0 == detail::safe_strcmp(key, "audio_filepath")) {
        entry.audio_filepath = parser.GetString();
      } else if (0 == detail::safe_strcmp(key, "duration")) {
        entry.duration = parser.GetDouble();
      } else if (0 == detail::safe_strcmp(key, "offset")) {
        entry.offset = parser.GetDouble();
        DALI_WARN("Handing of ``offset`` is not yet implemented and will be ignored.");
      } else if (0 == detail::safe_strcmp(key, "text")) {
        entry.text = parser.GetString();
      } else {
        parser.SkipValue();
      }
    }
    if (entry.audio_filepath.empty()) {
      DALI_WARN(make_string("Skipping manifest line without an audio filepath: ", line));
      continue;
    }

    if ((max_duration > 0.0f && entry.duration > max_duration) ||
        (min_duration > 0.0f && entry.duration < min_duration)) {
      continue;  // skipping sample
    }

    entries.emplace_back(std::move(entry));
  }
}

}  // namespace detail

void NemoAsrLoader::PrepareMetadataImpl() {
  std::ifstream fstream(manifest_filepath_);
  DALI_ENFORCE(fstream,
               make_string("Could not open NEMO ASR manifest file: \"", manifest_filepath_, "\""));
  detail::ParseManifest(entries_, fstream, max_duration_);

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

  // Ignoring copy_read_data_, Sharing data is not supported with this loader
  // TODO(janton): do not create a new decoder each time (?)
  using DecoderType = int16_t;

  GenericAudioDecoder<DecoderType> decoder;
  sample.audio_meta = decoder.OpenFromFile(entry.audio_filepath);
  assert(sample.audio_meta.channels_interleaved);  // it's always true

  sample.audio.set_type(TypeTable::GetTypeInfo(dtype_));
  auto shape = DecodedAudioShape(sample.audio_meta, sample_rate_, downmix_);
  assert(shape.size() > 0);
  sample.audio.Resize(shape);

  bool should_resample = sample_rate_ > 0 && sample.audio_meta.sample_rate != sample_rate_;
  bool should_downmix = sample.audio_meta.channels > 1 && downmix_;

  int64_t decode_scratch_sz = 0;
  int64_t resample_scratch_sz = 0;
  if (should_resample || should_downmix || dtype_ != DALI_INT16)
    decode_scratch_sz = sample.audio_meta.length * sample.audio_meta.channels;

  // resample scratch is used to prepare a single or multiple (depending if
  // downmixing is needed) channel float input, required by the resampling
  // kernel
  int64_t out_channels = should_downmix ? 1 : sample.audio_meta.channels;
  if (should_resample)
    resample_scratch_sz = sample.audio_meta.length * out_channels;

  int64_t total_scratch_sz =
      decode_scratch_sz * sizeof(DecoderType) + resample_scratch_sz * sizeof(float);
  scratch_.set_type(TypeTable::GetTypeInfo(DALI_UINT8));
  scratch_.Resize({total_scratch_sz});
  uint8_t* scratch_mem = scratch_.mutable_data<uint8_t>();

  span<DecoderType> decoder_scratch_mem(reinterpret_cast<DecoderType *>(scratch_mem),
                                        decode_scratch_sz);
  span<float> resample_scratch_mem(
        reinterpret_cast<float *>(scratch_mem + decode_scratch_sz * sizeof(DecoderType)),
        resample_scratch_sz);
  TYPE_SWITCH(dtype_, type2id, OutType, (float, int16_t), (
    DecodeAudio(view<OutType>(sample.audio), decoder, sample.audio_meta, resampler_,
            decoder_scratch_mem, resample_scratch_mem, sample_rate_, downmix_,
            entry.audio_filepath.c_str());
  ), DALI_FAIL(make_string("Unsupported type: ", dtype_)));  // NOLINT

  decoder.Close();
  sample.text = entry.text;
}

Index NemoAsrLoader::SizeImpl() {
  return entries_.size();
}
}  // namespace dali
