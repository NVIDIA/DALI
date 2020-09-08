// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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
#include <memory>
#include <istream>
#include <string>
#include <vector>
#include <utility>

#include "dali/operators/reader/loader/file_label_loader.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"
#include "dali/operators/decoder/audio/audio_decoder.h"
#include "dali/kernels/signal/resampling.h"

namespace dali {

struct NemoAsrEntry {
  std::string audio_filepath;
  float duration = 0.0;  // in seconds
  float offset = 0.0;  // in seconds, optional
  std::string text;  // transcription
};

struct AsrSample {
  Tensor<CPUBackend> audio;
  std::string text;
  AudioMetadata audio_meta;
};

namespace detail {

DLL_PUBLIC void ParseManifest(std::vector<NemoAsrEntry> &entries, std::istream &manifest_file);

}  // namespace detail

class DLL_PUBLIC NemoAsrLoader : public Loader<CPUBackend, AsrSample> {
 public:
  explicit inline NemoAsrLoader(const OpSpec &spec)
      : Loader<CPUBackend, AsrSample>(spec),
        manifest_filepath_(spec.GetArgument<std::string>("manifest_filepath")),
        shuffle_after_epoch_(spec.GetArgument<bool>("shuffle_after_epoch")),
        sample_rate_(spec.GetArgument<float>("sample_rate")),
        quality_(spec.GetArgument<float>("quality")),
        downmix_(spec.GetArgument<bool>("downmix")),
        dtype_(spec.GetArgument<DALIDataType>("dtype")) {
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
  std::string manifest_filepath_;
  std::vector<NemoAsrEntry> entries_;

  Tensor<CPUBackend> scratch_;

  bool shuffle_after_epoch_;
  Index current_index_ = 0;
  int current_epoch_ = 0;

  float sample_rate_;
  float quality_;
  bool downmix_;
  DALIDataType dtype_;

  kernels::signal::resampling::Resampler resampler_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_NEMO_ASR_LOADER_H_
