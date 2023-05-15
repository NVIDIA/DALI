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
#include "dali/core/cuda_event.h"
#include "dali/core/cuda_stream_pool.h"
#include "dali/core/mm/memory.h"
#include "dali/pipeline/data/types.h"
#include "dali/operators/reader/loader/cufile_loader.h"
#include "dali/operators/reader/loader/numpy_loader.h"
#include "dali/util/cufile.h"
#include "dali/operators/reader/gds_mem.h"

namespace dali {

struct NumpyFileWrapperGPU {
  std::string filename;
  bool fortran_order = false;
  int64_t data_offset = 0;
  TensorShape<> shape;
  DALIDataType type;
  DALIMeta meta;
  int source_sample_idx = -1;

  std::unique_ptr<CUFileStream> file_stream_;
  bool read_ahead = false;

  void ReadHeader(detail::NumpyHeaderCache &cache);

  void ReadRawChunk(void* buffer, size_t bytes, Index buffer_offset, Index offset);

  void Reopen();

  const TensorShape<>& get_shape() const {
    return shape;
  }

  DALIDataType get_type() const {
    return type;
  }

  const DALIMeta& get_meta() const {
    return meta;
  }
};


class NumpyLoaderGPU : public CUFileLoader<NumpyFileWrapperGPU> {
 public:
  using CUFileLoader<NumpyFileWrapperGPU>::CUFileLoader;

  void PrepareEmpty(NumpyFileWrapperGPU& tensor) override;
  void ReadSample(NumpyFileWrapperGPU& tensor) override;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_LOADER_NUMPY_LOADER_GPU_H_
