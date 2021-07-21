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

#include <string>
#include <numeric>
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/operators/decoder/audio/generic_decoder.h"
#include "dali/operators/decoder/audio/audio_decoder_impl.h"
#include "dali/operators/reader/loader/nemo_asr_loader.h"
#include "dali/operators/reader/loader/utils.h"
#include "dali/pipeline/data/views.h"
#include "dali/core/tensor_shape_print.h"
#include "dali/core/static_switch.h"
#include "dali/pipeline/util/lookahead_parser.h"
#include "dali/util/file.h"

namespace dali {

namespace detail {

void ParseManifest(std::vector<NemoAsrEntry> &entries, std::istream& manifest_file,
                   double min_duration, double max_duration, bool read_text) {
  std::string line;
  int64_t index = 0;
  while (std::getline(manifest_file, line)) {
    detail::LookaheadParser parser(const_cast<char*>(line.c_str()));
    if (parser.PeekType() != kObjectType) {
      DALI_WARN(make_string("Skipping invalid manifest line: ", line));
      continue;
    }
    parser.EnterObject();
    NemoAsrEntry entry;
    entry.index = index;
    while (const char* key = parser.NextObjectKey()) {
      if (0 == std::strcmp(key, "audio_filepath")) {
        entry.audio_filepath = parser.GetString();
      } else if (0 == std::strcmp(key, "duration")) {
        entry.duration = parser.GetDouble();
      } else if (0 == std::strcmp(key, "offset")) {
        entry.offset = parser.GetDouble();
      } else if (read_text && 0 == std::strcmp(key, "text")) {
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
    index++;
  }
}

}  // namespace detail

void NemoAsrLoader::PrepareMetadataImpl() {
  for (auto &manifest_filepath : manifest_filepaths_) {
    std::ifstream fstream(manifest_filepath);
    DALI_ENFORCE(fstream,
                 make_string("Could not open NEMO ASR manifest file: \"", manifest_filepath, "\""));
    detail::ParseManifest(entries_, fstream, min_duration_, max_duration_, read_text_);
  }
  shuffled_indices_.resize(entries_.size());
  std::iota(shuffled_indices_.begin(), shuffled_indices_.end(), 0);

  DALI_ENFORCE(SizeImpl() > 0, "No files found.");
  if (shuffle_) {
    // seeded with hardcoded value to get
    // the same sequence on every shard
    std::mt19937 g(kDaliDataloaderSeed);
    std::shuffle(shuffled_indices_.begin(), shuffled_indices_.end(), g);
  }
  Reset(true);
}

void NemoAsrLoader::Reset(bool wrap_to_shard) {
  current_index_ = wrap_to_shard ? start_index(shard_id_, num_shards_, SizeImpl()) : 0;
  current_epoch_++;

  if (shuffle_after_epoch_) {
    std::mt19937 g(kDaliDataloaderSeed + current_epoch_);
    std::shuffle(shuffled_indices_.begin(), shuffled_indices_.end(), g);
  }
}

void NemoAsrLoader::PrepareEmpty(AsrSample &sample) {
  sample = {};
}

template <typename OutputType>
void NemoAsrLoader::ReadAudio(Tensor<CPUBackend> &audio,
                              const AudioMetadata &audio_meta,
                              const NemoAsrEntry &entry,
                              AudioDecoderBase &decoder,
                              std::vector<float> &decode_scratch,
                              std::vector<float> &resample_scratch) {
  bool should_resample = sample_rate_ > 0 && audio_meta.sample_rate != sample_rate_;
  bool should_downmix = audio_meta.channels > 1 && downmix_;

  int64_t decode_scratch_sz = 0;
  if (should_resample || should_downmix)
    decode_scratch_sz = audio_meta.length * audio_meta.channels;
  decode_scratch.resize(decode_scratch_sz);

  // resample scratch is used to prepare a single or multiple (depending if
  // downmixing is needed) channel float input, required by the resampling
  // kernel
  int64_t resample_scratch_sz = 0;
  if (should_resample)
    resample_scratch_sz =
        should_downmix ? audio_meta.length : audio_meta.length * audio_meta.channels;
  resample_scratch.resize(resample_scratch_sz);

  DecodeAudio<OutputType>(
    view<OutputType>(audio), decoder, audio_meta, resampler_,
    {decode_scratch.data(), decode_scratch_sz},
    {resample_scratch.data(), resample_scratch_sz},
    sample_rate_, downmix_,
    entry.audio_filepath.c_str());
}

void NemoAsrLoader::ReadSample(AsrSample& sample) {
  auto &entry = entries_[shuffled_indices_[current_index_]];

  // handle wrap-around
  ++current_index_;
  MoveToNextShard(current_index_);

  // metadata info
  sample.index_ = entry.index;
  sample.text_ = entry.text;

  // Ignoring copy_read_data_, Sharing data is not supported with this loader

  bool use_resampling = sample_rate_ > 0;
  sample.decoder_ = make_generic_audio_decoder();

  auto &meta = sample.audio_meta_ = sample.decoder().OpenFromFile(entry.audio_filepath);
  assert(meta.channels_interleaved);  // it's always true

  int64_t offset, length;
  std::tie(offset, length) =
      ProcessOffsetAndLength(meta, entry.offset, entry.duration);
  assert(0 < length && length <= meta.length && "Unexpected length");
  meta.length = length;

  sample.shape_ = DecodedAudioShape(meta, sample_rate_, downmix_);
  assert(sample.shape_.size() > 0);
  sample.decoder().Close();  // avoid keeping too many files open at the same time.

  TYPE_SWITCH(dtype_, type2id, OutputType, (int16_t, int32_t, float), (
    // Audio decoding will be run in the prefetch function, once the batch is formed
    sample.decode_f_ = [this, &sample, &entry, offset](Tensor<CPUBackend> &audio, int tid) {
      sample.decoder().OpenFromFile(entry.audio_filepath);
      if (offset > 0)
        sample.decoder().SeekFrames(offset);
      ReadAudio<OutputType>(
        audio, sample.audio_meta_, entry, sample.decoder(),
        decode_scratch_[tid], resample_scratch_[tid]);
      sample.decoder().Close();
    };
  ), (  // NOLINT
    DALI_FAIL(make_string("Unsupported output type: ", dtype_,
                          ". Supported types are int16, int32, and float."));
  ));  // NOLINT
}

Index NemoAsrLoader::SizeImpl() {
  return entries_.size();
}
}  // namespace dali
