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

#ifndef DALI_OPERATORS_READER_LOADER_NUMPY_LOADER_GPU_H_
#define DALI_OPERATORS_READER_LOADER_NUMPY_LOADER_GPU_H_

#include <dirent.h>
#include <sys/stat.h>
#include <errno.h>

#include <fstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>
#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <mutex>

#include "dali/core/common.h"
#include "dali/pipeline/data/types.h"
#include "dali/operators/reader/loader/cufile_loader.h"
#include "dali/operators/reader/loader/numpy_loader.h"
#include "dali/util/cufile.h"

namespace dali {

struct NumpyFileWrapperGPU {
  std::string filename;
  bool fortran_order;
  TensorShape<> shape;
  TypeInfo type;
  DALIMeta meta;

  std::function<void(void)> read_meta_f;
  std::function<void(void* buffer, Index offset, size_t total_size)> read_sample_f;
  std::unique_ptr<CUFileStream> file_stream;

  const TensorShape<>& get_shape() const {
    return shape;
  }

  const TypeInfo& get_type() const {
    return type;
  }

  const DALIMeta& get_meta() const {
    return meta;
  }
};

class NumpyLoaderGPU : public CUFileLoader<NumpyFileWrapperGPU> {
 public:
  explicit inline NumpyLoaderGPU(const OpSpec& spec, vector<std::string> files = {},
                                 bool shuffle_after_epoch = false)
      : CUFileLoader(spec, files, shuffle_after_epoch),
        register_buffers_(false),
        header_cache_(spec.GetArgument<bool>("cache_header_information")) {}

  ~NumpyLoaderGPU() override {
    // set device
    DeviceGuard g(device_id_);

    // clean up buffers
    for (auto it = reg_buff_.begin(); it != reg_buff_.end(); ++it) {
      cuFileBufDeregister(it->first);
    }
    reg_buff_.clear();
  }

  void PrepareEmpty(NumpyFileWrapperGPU& tensor) override;
  void ReadSample(NumpyFileWrapperGPU& tensor) override;

 protected:
  // register input tensor
  void RegisterBuffer(void *buffer, size_t total_size);

  // read the full sample
  void ReadSampleHelper(CUFileStream *file,
                        void *buffer, Index offset, size_t total_size);

  // do we want to register device buffers:
  bool register_buffers_;

  // registered buffer addresses
  std::mutex reg_mutex_;
  std::map<uint8_t*, size_t> reg_buff_;

  detail::NumpyHeaderCache header_cache_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_NUMPY_LOADER_GPU_H_
