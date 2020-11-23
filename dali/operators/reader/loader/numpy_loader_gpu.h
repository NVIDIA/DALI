// Copyright (c) 2017, NVIDIA CORPORATION. All rights reserved.
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
#include <regex>
#include <memory>
#include <set>
#include <mutex>

#include "dali/core/common.h"
#include "dali/pipeline/data/types.h"
#include "dali/operators/reader/loader/cufile_loader.h"
#include "dali/operators/reader/loader/numpy_loader.h"
#include "dali/util/cufile.h"

namespace dali {

class NumpyLoaderGPU : public CUFileLoader {
 public:
  explicit inline NumpyLoaderGPU(
    const OpSpec& spec,
    vector<std::string> images = std::vector<std::string>(),
    bool shuffle_after_epoch = false) :
      CUFileLoader(spec, images, shuffle_after_epoch),
    register_buffers_(spec.GetArgument<bool>("register_buffers")),
    header_regex_(R"###(^\{'descr': \'(.*?)\', 'fortran_order': (.*?), 'shape': \((.*?)\), \})###"),
    cache_headers_(spec.GetArgument<bool>("cache_header_information")) {
    // set device
    DeviceGuard g(device_id_);
  }

  ~NumpyLoaderGPU() override {
    // set device
    DeviceGuard g(device_id_);

    // clean up buffers
    for (auto it = reg_buff_.begin(); it != reg_buff_.end(); ++it) {
      cuFileBufDeregister(it->first);
    }
    reg_buff_.clear();
  }

  // we want to make it possible to override this function as well
  void ReadSample(ImageFileWrapperGPU& tensor) override;

 protected:
  // parser function, only for internal use
  std::unique_ptr<CUFileStream>
    ParseHeader(std::unique_ptr<CUFileStream> file, NumpyParseTarget& target);

  // register input tensor
  void RegisterTensor(void *buffer, size_t total_size);

  // read the full sample
  void ReadSampleHelper(CUFileStream *file, ImageFileWrapperGPU& imfile,
                        void *buffer, Index offset, size_t total_size);

  // do we want to register device buffers:
  bool register_buffers_;

  // registered buffer addresses
  std::mutex reg_mutex_;
  std::map<uint8_t*, size_t> reg_buff_;

  // regex search string
  const std::regex header_regex_;

  // helper for header caching
  std::mutex cache_mutex_;
  bool cache_headers_;
  std::map<string, NumpyParseTarget> header_cache_;

  // temporary tensor for 2-stage IO
  Tensor<GPUBackend> io_tensor_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_NUMPY_LOADER_GPU_H_
