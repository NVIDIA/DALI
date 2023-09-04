// Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
#include <errno.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "dali/core/common.h"
#include "dali/operators/reader/loader/file_loader.h"
#include "dali/operators/reader/loader/loader.h"
#include "dali/operators/reader/loader/utils.h"
#include "dali/util/cufile.h"

namespace dali {

template <typename Target>
class CUFileLoader : public FileLoader<GPUBackend, Target, CUFileStream> {
 public:
  explicit CUFileLoader(const OpSpec& spec, vector<std::string> images = {},
                        bool shuffle_after_epoch = false)
      : FileLoader<GPUBackend, Target, CUFileStream>(spec, shuffle_after_epoch) {
  }

  ~CUFileLoader() {
    /*
     * As this class keeps the CUFileDriverHandle open as long as it lives we need to make sure
     * when it is closed there is no more resources that may use the cuFile. In this case
     * last_sample_ptr_tmp, sample_buffer_ and empty_tensors_ when destroyed still uses
     * cuFileDeregister functions, so instead of letting them to be cleared by Loader class when
     * cuFile is no longer accesible we need to do that here.
     */
    this->last_sample_ptr_tmp.reset();
    this->sample_buffer_.clear();
    this->empty_tensors_.clear();
  }

 private:
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_CUFILE_LOADER_H_
