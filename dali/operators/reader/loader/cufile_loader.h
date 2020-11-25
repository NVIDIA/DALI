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

#ifndef DALI_OPERATORS_READER_LOADER_CUFILE_LOADER_H_
#define DALI_OPERATORS_READER_LOADER_CUFILE_LOADER_H_

#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>

#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <algorithm>
#include <memory>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/loader.h"
#include "dali/operators/reader/loader/file_loader.h"
#include "dali/util/cufile_helper.h"
#include "dali/util/cufile.h"

namespace dali {

struct ImageFileWrapperGPU {
  Tensor<GPUBackend> image;
  std::string filename;
  bool transpose_fortan_order;
  TensorShape<> shape;
  TypeInfo type_info;
  std::unique_ptr<CUFileStream> file_stream;
  std::function<void(void)> read_meta_f;
  std::function<void(void *buffer, Index offset, size_t total_size)> read_sample_f;
};

class CUFileLoader : public Loader<GPUBackend, ImageFileWrapperGPU> {
 public:
  explicit CUFileLoader(
    const OpSpec& spec,
    vector<std::string> images = std::vector<std::string>(),
    bool shuffle_after_epoch = false);

  ~CUFileLoader() {
    /*
     * As this class keeps the CUFileDriverHandle open as long as it lives we need to make sure
     * when it is closed there is no more resources that may use the cuFile. In this case
     * last_sample_ptr_tmp, sample_buffer_ and empty_tensors_ when destroyed still uses
     * cuFileDeregister functions, so instead of letting them to be cleared by Loader class when
     * cuFile is no longer accesible we need to do that here.
     */
    last_sample_ptr_tmp.reset();
    sample_buffer_.clear();
    empty_tensors_.clear();
  }

  void PrepareEmpty(ImageFileWrapperGPU& tensor) override;
  // we want to make it possible to override this function as well
  void ReadSample(ImageFileWrapperGPU& tensor) override;

 protected:
  Index SizeImpl() override;

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
    DALI_ENFORCE(Size() > 0, "No files found.");

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
      current_index_ = start_index(shard_id_, num_shards_, Size());
    } else {
      current_index_ = 0;
    }

    current_epoch_++;

    if (shuffle_after_epoch_) {
      std::mt19937 g(kDaliDataloaderSeed + current_epoch_);
      std::shuffle(images_.begin(), images_.end(), g);
    }
  }

  using Loader<GPUBackend, ImageFileWrapperGPU >::shard_id_;
  using Loader<GPUBackend, ImageFileWrapperGPU >::num_shards_;

  std::shared_ptr<cufile::CUFileDriverHandle> d_;
  string file_root_, file_list_, file_filter_;
  vector<std::string> images_;

  bool has_files_arg_     = false;
  bool has_file_list_arg_ = false;
  bool has_file_root_arg_ = false;

  bool shuffle_after_epoch_;
  Index current_index_;
  int current_epoch_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_CUFILE_LOADER_H_
