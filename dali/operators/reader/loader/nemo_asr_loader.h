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

#ifndef DALI_OPERATORS_READER_LOADER_NEMO_ASR_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_NEMO_ASR_LOADER_H_

#include <algorithm>
#include <future>
#include <istream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/kernels/signal/resampling.h"
#include "dali/operators/decoder/audio/audio_decoder.h"
#include "dali/operators/decoder/audio/audio_decoder_impl.h"
#include "dali/operators/reader/loader/file_label_loader.h"
#include "dali/pipeline/util/thread_pool.h"

namespace dali {

static constexpr double kDefaultDuration = -1.0;

struct NemoAsrEntry {
  std::string audio_filepath;
  double duration = kDefaultDuration;  // in seconds, optional
  double offset = 0.0;  // in seconds, optional
  int64_t index = -1;
  std::string text;  // transcription
};

class AsrSample {
 public:
  int64_t index() const {
    return index_;
  }

  const std::string& text() const {
    return text_;
  }

  const AudioMetadata &audio_meta() const {
    return audio_meta_;
  }

  const std::string& audio_filepath() const {
    return audio_filepath_;
  }

  const TensorShape<>& shape() const {
    return shape_;
  }

  void decode_audio(Tensor<CPUBackend>& audio, int tid) {
    decode_f_(audio, tid);
  }

  friend class NemoAsrLoader;

  AsrSample() = default;
  AsrSample(AsrSample &&) = default;
  AsrSample& operator=(AsrSample&&) = default;

 private:
  AudioDecoderBase &decoder() {
    assert(decoder_);
    return *decoder_;
  }

  int64_t index_ = 0;
  std::string text_;
  AudioMetadata audio_meta_;
  std::string audio_filepath_;  // for tensor metadata purposes
  TensorShape<> shape_;

  std::function<void(Tensor<CPUBackend>&, int)> decode_f_;
  std::unique_ptr<AudioDecoderBase> decoder_;
};

namespace detail {

/**
 * @brief Parses the contents of a manifest file and populates a vector of NemoAsrEntry
 * @param min_duration Minimum audio duration, in seconds. Shorter samples will be filtered out.
 * @param max_duration Maximum audio duration, in seconds. Longer samples will be filtered out.
 * @param read_text If True, the parser will read the text transcript from the manifest.
 *                  If False, the text field is ignored.
 */
DLL_PUBLIC void ParseManifest(std::vector<NemoAsrEntry> &entries, std::istream &manifest_file,
                              double min_duration = kDefaultDuration,
                              double max_duration = kDefaultDuration,
                              bool read_text = true);

}  // namespace detail

class DLL_PUBLIC NemoAsrLoader : public Loader<CPUBackend, AsrSample> {
 public:
  explicit inline NemoAsrLoader(const OpSpec &spec)
      : Loader<CPUBackend, AsrSample>(spec),
        manifest_filepaths_(spec.GetRepeatedArgument<std::string>("manifest_filepaths")),
        shuffle_after_epoch_(spec.GetArgument<bool>("shuffle_after_epoch")),
        sample_rate_(spec.GetArgument<float>("sample_rate")),
        quality_(spec.GetArgument<float>("quality")),
        downmix_(spec.GetArgument<bool>("downmix")),
        dtype_(spec.GetArgument<DALIDataType>("dtype")),
        min_duration_(spec.GetArgument<float>("min_duration")),
        max_duration_(spec.GetArgument<float>("max_duration")),
        read_text_(spec.GetArgument<bool>("read_text")),
        num_threads_(std::max(1, spec.GetArgument<int>("num_threads"))),
        decode_scratch_(num_threads_),
        resample_scratch_(num_threads_) {
    DALI_ENFORCE(!manifest_filepaths_.empty(), "``manifest_filepaths`` can not be empty");
    /*
     * Those options are mutually exclusive as `shuffle_after_epoch` will make every shard looks
     * differently after each epoch so coexistence with `stick_to_shard` doesn't make any sense
     * Still when `shuffle_after_epoch` we will set `stick_to_shard` internally in the FileLoader so
     * all DALI instances will do shuffling after each epoch
     */
    if (shuffle_after_epoch_ && stick_to_shard_)
      DALI_FAIL("`shuffle_after_epoch` and `stick_to_shard` can't be provided together");
    if (shuffle_after_epoch_ && shuffle_)
      DALI_FAIL("`shuffle_after_epoch` and `random_shuffle` can't be provided together");
    /*
     * Imply `stick_to_shard` from  `shuffle_after_epoch`
     */
    if (shuffle_after_epoch_)
      stick_to_shard_ = true;

    double q = quality_;
    DALI_ENFORCE(q >= 0 && q <= 100, "Resampling quality must be in [0..100] range");
    // this should give 3 lobes for q = 0, 16 lobes for q = 50 and 64 lobes for q = 100
    int lobes = std::round(0.007 * q * q - 0.09 * q + 3);
    resampler_.Initialize(lobes, lobes * 64 + 1);
  }

  ~NemoAsrLoader() override = default;
  void PrepareEmpty(AsrSample &sample) override;
  void ReadSample(AsrSample& sample) override;

 protected:
  void PrepareMetadataImpl() override;
  Index SizeImpl() override;
  void Reset(bool wrap_to_shard) override;

 private:
  template <typename OutputType>
  void ReadAudio(Tensor<CPUBackend> &audio,
                 const AudioMetadata &audio_meta,
                 const NemoAsrEntry &entry,
                 AudioDecoderBase &decoder,
                 std::vector<float> &decode_scratch,
                 std::vector<float> &resample_scratch);

  std::vector<std::string> manifest_filepaths_;
  std::vector<NemoAsrEntry> entries_;
  std::vector<size_t> shuffled_indices_;

  bool shuffle_after_epoch_;
  Index current_index_ = 0;
  int current_epoch_ = 0;

  float sample_rate_;
  float quality_;
  bool downmix_;
  DALIDataType dtype_;
  double min_duration_;
  double max_duration_;
  bool read_text_;
  int num_threads_;
  kernels::signal::resampling::Resampler resampler_;
  std::vector<std::vector<float>> decode_scratch_;
  std::vector<std::vector<float>> resample_scratch_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_NEMO_ASR_LOADER_H_
