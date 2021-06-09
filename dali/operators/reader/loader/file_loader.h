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

#ifndef DALI_OPERATORS_READER_LOADER_FILE_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_FILE_LOADER_H_

#include <dirent.h>
#include <errno.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/filesystem.h"
#include "dali/operators/reader/loader/loader.h"
#include "dali/operators/reader/loader/utils.h"
#include "dali/util/file.h"

namespace dali {

struct ImageFileWrapper {
  Tensor<CPUBackend> image;
  std::string filename;
  // some field for auxiliary info to pass to the reader
  bool fortran_order;
};

template <typename Backend = CPUBackend, typename Target = ImageFileWrapper,
          typename InputStream = FileStream>
class FileLoader : public Loader<Backend, Target> {
 public:
  explicit inline FileLoader(const OpSpec &spec, bool shuffle_after_epoch = false)
      : Loader<Backend, Target>(spec),
        file_filter_(spec.GetArgument<string>("file_filter")),
        shuffle_after_epoch_(shuffle_after_epoch),
        current_index_(0),
        current_epoch_(0) {
    vector<string> files;

    has_files_arg_ = spec.TryGetRepeatedArgument(files, "files");
    has_file_list_arg_ = spec.TryGetArgument(file_list_, "file_list");
    has_file_root_arg_ = spec.TryGetArgument(file_root_, "file_root");

    DALI_ENFORCE(has_file_root_arg_ || has_files_arg_ || has_file_list_arg_,
                 "``file_root`` argument is required when not using ``files`` or ``file_list``.");

    DALI_ENFORCE(has_files_arg_ + has_file_list_arg_ <= 1,
                 "File paths can be provided through ``files`` or ``file_list`` but not both.");

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
      images_ = std::move(files);
    }

    /*
     * Those options are mutually exclusive as `shuffle_after_epoch` will make every shard looks
     * differently after each epoch so coexistence with `stick_to_shard` doesn't make any sense
     * Still when `shuffle_after_epoch` we will set `stick_to_shard` internally in the FileLoader so
     * all DALI instances will do shuffling after each epoch
     */
    DALI_ENFORCE(!(shuffle_after_epoch_ && stick_to_shard_),
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
      mmap_reserver_ =
          typename InputStream::MappingReserver(static_cast<unsigned int>(initial_buffer_fill_));
    }
    copy_read_data_ = dont_use_mmap_ || !mmap_reserver_.CanShareMappedData();
  }

  void PrepareEmpty(Target &image_file) override {
    PrepareEmptyTensor(image_file.image);
    image_file.filename = "";
  }

  void ReadSample(Target &imfile) override {
    auto image_file = images_[current_index_++];

    // handle wrap-around
    MoveToNextShard(current_index_);

    // metadata info
    DALIMeta meta;
    meta.SetSourceInfo(image_file);
    meta.SetSkipSample(false);

    // if image is cached, skip loading
    if (ShouldSkipImage(image_file)) {
      meta.SetSkipSample(true);
      imfile.image.Reset();
      imfile.image.SetMeta(meta);
      imfile.image.set_type(TypeInfo::Create<uint8_t>());
      imfile.image.Resize({0});
      imfile.filename = "";
      return;
    }

    auto current_image = InputStream::Open(filesystem::join_path(file_root_, image_file),
                                           read_ahead_, !copy_read_data_);
    Index image_size = current_image->Size();

    if (copy_read_data_) {
      if (imfile.image.shares_data()) {
        imfile.image.Reset();
      }
      imfile.image.Resize({image_size});
      // copy the image
      Index ret = current_image->Read(imfile.image.template mutable_data<uint8_t>(), image_size);
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
    imfile.filename = filesystem::join_path(file_root_, image_file);
  }

 protected:
  Index SizeImpl() override {
    return static_cast<Index>(images_.size());
  }

  void PrepareMetadataImpl() override {
    if (images_.empty()) {
      if (!has_files_arg_ && !has_file_list_arg_) {
        images_ = filesystem::traverse_directories(file_root_, file_filter_);
      } else if (has_file_list_arg_) {
        // load paths from list
        std::ifstream s(file_list_);
        DALI_ENFORCE(s.is_open(), "Cannot open: " + file_list_);

        vector<char> line_buf(16 << 10);  // 16 kB should be more than enough for a line
        char *line = line_buf.data();
        while (s.getline(line, line_buf.size())) {
          if (line[0])  // skip empty lines
            images_.emplace_back(line);
        }
        DALI_ENFORCE(s.eof(), "Wrong format of file_list: " + file_list_);
      }
    }
    DALI_ENFORCE(SizeImpl() > 0, "No files found.");

    if (shuffle_) {
      // seeded with hardcoded value to get
      // the same sequence on every shard
      std::mt19937 g(kDaliDataloaderSeed);
      std::shuffle(images_.begin(), images_.end(), g);
    }
    Reset(true);
  }

  void Reset(bool wrap_to_shard) override {
    if (wrap_to_shard) {
      current_index_ = start_index(shard_id_, num_shards_, SizeImpl());
    } else {
      current_index_ = 0;
    }

    current_epoch_++;

    if (shuffle_after_epoch_) {
      std::mt19937 g(kDaliDataloaderSeed + current_epoch_);
      std::shuffle(images_.begin(), images_.end(), g);
    }
  }

  using Loader<Backend, Target>::shard_id_;
  using Loader<Backend, Target>::num_shards_;
  using Loader<Backend, Target>::stick_to_shard_;
  using Loader<Backend, Target>::shuffle_;
  using Loader<Backend, Target>::dont_use_mmap_;
  using Loader<Backend, Target>::initial_buffer_fill_;
  using Loader<Backend, Target>::copy_read_data_;
  using Loader<Backend, Target>::read_ahead_;
  using Loader<Backend, Target>::MoveToNextShard;
  using Loader<Backend, Target>::ShouldSkipImage;
  using Loader<Backend, Target>::Size;
  using Loader<Backend, Target>::PrepareEmptyTensor;

  string file_list_, file_root_, file_filter_;
  vector<std::string> images_;

  bool has_files_arg_ = false;
  bool has_file_list_arg_ = false;
  bool has_file_root_arg_ = false;

  bool shuffle_after_epoch_;
  Index current_index_;
  int current_epoch_;
  typename InputStream::MappingReserver mmap_reserver_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_FILE_LOADER_H_
