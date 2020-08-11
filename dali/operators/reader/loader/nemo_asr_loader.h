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
#include <string>
#include <vector>
#include <utility>

#include "dali/operators/reader/loader/file_label_loader.h"
#include "dali/core/common.h"
#include "dali/core/error_handling.h"

namespace dali {

struct NemoAsrEntry {
  std::string audio_filepath;
  float duration;  // in seconds
  float offset;  // in seconds, optional
  std::string text;  // transcription
};

class NemoAsrLoader : public Loader<CPUBackend, NemoAsrEntry> {
 public:
  explicit inline NemoAsrLoader(const OpSpec &spec, const std::string &manifest_filepath,
                                bool shuffle_after_epoch = false)
      : Loader<CPUBackend, NemoAsrEntry>(spec),
        spec_(spec),
        manifest_filepath_(spec.GetArgument<std::string>("manifest_filepath")) {
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
  }

 protected:
  void PrepareMetadataImpl() override;
  Index SizeImpl() override;
  void Reset(bool wrap_to_shard) override;

 private:
  const OpSpec &spec_;
  std::string manifest_filepath_;
  std::vector<NemoAsrEntry> entries_;

  bool shuffle_after_epoch_;
  Index current_index_ = 0;
  int current_epoch_ = 0;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_NEMO_ASR_LOADER_H_
