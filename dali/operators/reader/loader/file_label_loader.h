// Copyright (c) 2017-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_LOADER_FILE_LABEL_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_FILE_LABEL_LOADER_H_

#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/discover_files.h"
#include "dali/operators/reader/loader/filesystem.h"
#include "dali/operators/reader/loader/loader.h"
#include "dali/util/file.h"

namespace dali {

struct ImageLabelWrapper {
  Tensor<CPUBackend> image;
  int label;

  // Deferred file read: If not null, means image was not read yet
  std::unique_ptr<FileStream> file_stream;
};



template<bool supports_checkpointing>
class DLL_PUBLIC FileLabelLoaderBase : public Loader<CPUBackend, ImageLabelWrapper,
                                                     supports_checkpointing> {
 public:
  using Base = Loader<CPUBackend, ImageLabelWrapper, supports_checkpointing>;
  explicit inline FileLabelLoaderBase(
    const OpSpec& spec,
    bool shuffle_after_epoch)
    : Base(spec),
      shuffle_after_epoch_(shuffle_after_epoch),
      current_index_(0),
      current_epoch_(0) {

    vector<string> files;
    vector<int> labels;

    has_files_arg_ = spec.TryGetRepeatedArgument(files, "files");
    has_labels_arg_ = spec.TryGetRepeatedArgument(labels, "labels");
    has_file_list_arg_ = spec.TryGetArgument(file_list_, "file_list");
    has_file_root_arg_ = spec.TryGetArgument(file_root_, "file_root");
    bool has_file_filters_arg =
      spec.TryGetRepeatedArgument(file_discovery_opts_.file_filters, "file_filters");
    bool has_dir_filters_arg =
      spec.TryGetRepeatedArgument(file_discovery_opts_.dir_filters, "dir_filters");

    // TODO(ksztenderski): CocoLoader inherits after FileLabelLoader and it doesn't work with
    // GetArgument.
    spec.TryGetArgument(file_discovery_opts_.case_sensitive_filter, "case_sensitive_filter");

    DALI_ENFORCE(has_file_root_arg_ || has_files_arg_ || has_file_list_arg_,
      "``file_root`` argument is required when not using ``files`` or ``file_list``.");

    DALI_ENFORCE(has_files_arg_ + has_file_list_arg_ <= 1,
      "File paths can be provided through ``files`` or ``file_list`` but not both.");

    DALI_ENFORCE(has_files_arg_ || !has_labels_arg_,
      "The argument ``labels`` is valid only when file paths "
      "are provided as `files` argument.");

    DALI_ENFORCE(!has_file_filters_arg || file_discovery_opts_.file_filters.size() > 0,
                 "``file_filters`` list cannot be empty.");
    DALI_ENFORCE(!has_dir_filters_arg || file_discovery_opts_.dir_filters.size() > 0,
                 "``dir_filters`` list cannot be empty.");

    if (has_file_list_arg_) {
      DALI_ENFORCE(!file_list_.empty(), "``file_list`` argument cannot be empty");
      if (!has_file_root_arg_) {
        auto idx = file_list_.rfind(filesystem::dir_sep);
        if (idx != string::npos) {
          file_root_ = file_list_.substr(0, idx);
        }
      }
    }

    if (has_files_arg_) {
      DALI_ENFORCE(files.size() > 0, "``files`` specified an empty list.");
      if (has_labels_arg_) {
        DALI_ENFORCE(files.size() == labels.size(), make_string("Provided ", labels.size(),
          " labels for ", files.size(), " files."));

        for (int i = 0, n = files.size(); i < n; i++)
          file_label_entries_.push_back({std::move(files[i]), labels[i]});
      } else {
          for (int i = 0, n = files.size(); i < n; i++)
            file_label_entries_.push_back({std::move(files[i]), i});
      }
    }

    /*
     * Those options are mutually exclusive as `shuffle_after_epoch` will make every shard looks differently
     * after each epoch so coexistence with `stick_to_shard` doesn't make any sense
     * Still when `shuffle_after_epoch` we will set `stick_to_shard` internally in the FileLabelLoader so all
     * DALI instances will do shuffling after each epoch
     */
    DALI_ENFORCE(!(shuffle_after_epoch_  && stick_to_shard_),
                  "shuffle_after_epoch and stick_to_shard cannot be both true");
    DALI_ENFORCE(!(shuffle_after_epoch_ && shuffle_),
                  "shuffle_after_epoch and random_shuffle cannot be both true");
    /*
     * Imply `stick_to_shard` from  `shuffle_after_epoch`
     */
    if (shuffle_after_epoch_) {
      stick_to_shard_ = true;
    }
    if (!dont_use_mmap_) {
      mmap_reserver_ = FileStream::MappingReserver(
                                  static_cast<unsigned int>(initial_buffer_fill_));
    }
    copy_read_data_ = dont_use_mmap_ || !mmap_reserver_.CanShareMappedData();
  }

  void PrepareEmpty(ImageLabelWrapper &tensor) override;
  void ReadSample(ImageLabelWrapper &tensor) override;

 protected:
  Index SizeImpl() override;

  void PrepareMetadataImpl() override {
    if (file_label_entries_.empty()) {
      if (!has_file_list_arg_ && !has_files_arg_) {
        file_label_entries_ = discover_files(file_root_, file_discovery_opts_);
      } else if (has_file_list_arg_) {
        // load (path, label) pairs from list
        std::ifstream s(file_list_);
        DALI_ENFORCE(s.is_open(), "Cannot open: " + file_list_);

        vector<char> line_buf(16 << 10);  // 16 kB should be more than enough for a line
        char *line = line_buf.data();
        for  (int n = 1; s.getline(line, line_buf.size()); n++) {
          // parse the line backwards:
          // - skip trailing whitespace
          // - consume digits
          // - skip whitespace between label and
          int i = strlen(line) - 1;

          for (; i >= 0 && isspace(line[i]); i--) {}  // skip trailing spaces

          int label_end = i + 1;

          if (i < 0)  // empty line - skip
            continue;

          for (; i >= 0 && isdigit(line[i]); i--) {}  // skip

          int label_start = i + 1;

          for (; i >= 0 && isspace(line[i]); i--) {}

          int name_end = i + 1;
          DALI_ENFORCE(name_end > 0 && name_end < label_start &&
                       label_start >= 2 && label_end > label_start,
                       make_string("Incorrect format of the list file \"",  file_list_, "\":", n,
                       " expected file name followed by a label; got: ", line));

          line[label_end] = 0;
          line[name_end] = 0;

          file_label_entries_.push_back({std::string(line), std::atoi(line + label_start)});
        }

        DALI_ENFORCE(s.eof(), "Wrong format of file_list: " + file_list_);
      }
    }
    DALI_ENFORCE(SizeImpl() > 0, "No files found.");

    if (shuffle_) {
      // seeded with hardcoded value to get
      // the same sequence on every shard
      std::mt19937 g(kDaliDataloaderSeed);
      std::shuffle(file_label_entries_.begin(), file_label_entries_.end(), g);
    }

    if (IsCheckpointingEnabled() && shuffle_after_epoch_) {
      // save initial order
      // DO not call std::move on file_label_entries_, even though it is restored in Reset
      // it may be needed if SizeImpl is called!
      backup_file_label_entries_ = file_label_entries_;
    }

    Reset(true);
  }

  void Skip() override {
    MoveToNextShard(++current_index_);
  }

  void Reset(bool wrap_to_shard) override {
    if (wrap_to_shard) {
      current_index_ = start_index(virtual_shard_id_, num_shards_, SizeImpl());
    } else {
      current_index_ = 0;
    }
    current_epoch_++;

    if (shuffle_after_epoch_) {
      if (IsCheckpointingEnabled()) {
        // With checkpointing enabled, dataset order must be easy to restore.
        // The shuffling is run with different seed every epoch, so this doesn't impact
        // the random distribution.
        file_label_entries_ = backup_file_label_entries_;
      }
      std::mt19937 g(kDaliDataloaderSeed + current_epoch_);
      std::shuffle(file_label_entries_.begin(), file_label_entries_.end(), g);
    }
  }

  void RestoreStateImpl(const LoaderStateSnapshot &state) override {
    current_epoch_ = state.current_epoch;
  }

  using Base::shard_id_;
  using Base::virtual_shard_id_;
  using Base::num_shards_;
  using Base::stick_to_shard_;
  using Base::shuffle_;
  using Base::dont_use_mmap_;
  using Base::initial_buffer_fill_;
  using Base::copy_read_data_;
  using Base::read_ahead_;
  using Base::IsCheckpointingEnabled;
  using Base::PrepareEmptyTensor;
  using Base::MoveToNextShard;
  using Base::ShouldSkipImage;

  string file_root_, file_list_;
  vector<FileLabelEntry> file_label_entries_;
  vector<FileLabelEntry> backup_file_label_entries_;
  FileDiscoveryOptions file_discovery_opts_;

  bool has_files_arg_ = false;
  bool has_labels_arg_ = false;
  bool has_file_list_arg_ = false;
  bool has_file_root_arg_ = false;

  bool shuffle_after_epoch_;
  Index current_index_;
  int current_epoch_;
  FileStream::MappingReserver mmap_reserver_;
};

using FileLabelLoader = FileLabelLoaderBase<true>;

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_FILE_LABEL_LOADER_H_
