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

std::string trim(const std::string& str,
                 const std::string& whitespace = " \t\f\n\r\v") {
  const auto str_begin = str.find_first_not_of(whitespace);
  if (str_begin == std::string::npos)
    return {};  // no content
  const auto str_end = str.find_last_not_of(whitespace);
  const auto str_len = str_end - str_begin + 1;
  return str.substr(str_begin, str_len);
}

std::string NormalizeText(const std::string& text) {
  // Remove trailing and leading whitespace
  auto norm_text = trim(text);
  // TODO(janton): Handle non-ascii
  std::transform(norm_text.begin(), norm_text.end(), norm_text.begin(),
    [](char c){ return std::tolower(c); });
  return norm_text;
}

void ParseManifest(std::vector<NemoAsrEntry> &entries, std::istream& manifest_file,
                   float min_duration, float max_duration, bool normalize_text) {
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
      if (0 == std::strcmp(key, "audio_filepath")) {
        entry.audio_filepath = parser.GetString();
      } else if (0 == std::strcmp(key, "duration")) {
        entry.duration = parser.GetDouble();
      } else if (0 == std::strcmp(key, "offset")) {
        entry.offset = parser.GetDouble();
        DALI_WARN("Handling of ``offset`` is not yet implemented and will be ignored.");
      } else if (0 == std::strcmp(key, "text")) {
        entry.text = parser.GetString();
        if (normalize_text)
          entry.text = NormalizeText(entry.text);
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
  for (auto &manifest_filepath : manifest_filepaths_) {
    std::ifstream fstream(manifest_filepath);
    DALI_ENFORCE(fstream,
                 make_string("Could not open NEMO ASR manifest file: \"", manifest_filepath, "\""));
    detail::ParseManifest(entries_, fstream, max_duration_, normalize_text_);
  }
  shuffled_indices_.resize(entries_.size());
  std::iota(shuffled_indices_.begin(), shuffled_indices_.end(), 0);

  DALI_ENFORCE(Size() > 0, "No files found.");
  if (shuffle_) {
    // seeded with hardcoded value to get
    // the same sequence on every shard
    std::mt19937 g(kDaliDataloaderSeed);
    std::shuffle(shuffled_indices_.begin(), shuffled_indices_.end(), g);
  }
  Reset(true);
}

void NemoAsrLoader::Reset(bool wrap_to_shard) {
  current_index_ = wrap_to_shard ? start_index(shard_id_, num_shards_, Size()) : 0;
  current_epoch_++;

  if (shuffle_after_epoch_) {
    std::mt19937 g(kDaliDataloaderSeed + current_epoch_);
    std::shuffle(shuffled_indices_.begin(), shuffled_indices_.end(), g);
  }
}

void NemoAsrLoader::PrepareEmpty(AsrSample &sample) {
  sample = {};
}

template <typename OutputType, typename DecoderOutputType>
void NemoAsrLoader::ReadAudio(Tensor<CPUBackend> &audio,
                              const AudioMetadata &audio_meta,
                              const NemoAsrEntry &entry,
                              AudioDecoderBase &decoder,
                              std::vector<uint8_t> &decode_scratch,
                              std::vector<float> &resample_scratch) {
  bool should_resample = sample_rate_ > 0 && audio_meta.sample_rate != sample_rate_;
  bool should_downmix = audio_meta.channels > 1 && downmix_;

  int64_t decode_scratch_sz = 0;
  int64_t resample_scratch_sz = 0;
  if (should_resample || should_downmix || dtype_ != DALI_INT16)
    decode_scratch_sz = audio_meta.length * audio_meta.channels;

  decode_scratch.resize(decode_scratch_sz * sizeof(DecoderOutputType));

  // resample scratch is used to prepare a single or multiple (depending if
  // downmixing is needed) channel float input, required by the resampling
  // kernel
  int64_t out_channels = should_downmix ? 1 : audio_meta.channels;
  if (should_resample)
    resample_scratch_sz = audio_meta.length * out_channels;

  resample_scratch.resize(resample_scratch_sz);

  DecodeAudio<OutputType, DecoderOutputType>(
    view<OutputType>(audio), decoder, audio_meta, resampler_,
    {reinterpret_cast<DecoderOutputType *>(decode_scratch.data()), decode_scratch_sz},
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
  sample.text_ = entry.text;

  // Ignoring copy_read_data_, Sharing data is not supported with this loader

  bool use_resampling = sample_rate_ > 0;
  DALIDataType decode_type = use_resampling ? DALI_FLOAT : dtype_;
  TYPE_SWITCH(dtype_, type2id, OutputType, (int16_t, int32_t, float), (
    TYPE_SWITCH(decode_type, type2id, DecoderOutputType, (int16_t, int32_t, float), (
      sample.decoder_ = std::make_unique<GenericAudioDecoder<DecoderOutputType>>();

      sample.audio_meta_ = sample.decoder().OpenFromFile(entry.audio_filepath);
      assert(sample.audio_meta_.channels_interleaved);  // it's always true

      sample.shape_ = DecodedAudioShape(sample.audio_meta(), sample_rate_, downmix_);
      assert(sample.shape_.size() > 0);

      // Audio decoding will be run in the prefetch function, once the batch is formed
      sample.decode_f_ = [this, &sample, &entry](Tensor<CPUBackend> &audio, int tid) {
        ReadAudio<OutputType, DecoderOutputType>(
          audio, sample.audio_meta_, entry, sample.decoder(),
          decode_scratch_[tid], resample_scratch_[tid]);
        sample.decoder().Close();
      };
    ), (  // NOLINT
      DALI_FAIL(make_string("Unsupported decoder output type: ", decode_type,
                            ". Supported types are int16, int32, and float."));
    ));  // NOLINT
  ), (  // NOLINT
    DALI_FAIL(make_string("Unsupported output type: ", dtype_,
                          ". Supported types are int16, int32, and float."));
  ));  // NOLINT
}

Index NemoAsrLoader::SizeImpl() {
  return entries_.size();
}
}  // namespace dali
