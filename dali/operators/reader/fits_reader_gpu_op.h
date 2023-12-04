// Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_FITS_READER_GPU_OP_H_
#define DALI_OPERATORS_READER_FITS_READER_GPU_OP_H_

#include <utility>

#include "dali/operators/reader/fits_reader_op.h"
#include "dali/operators/reader/loader/fits_loader_gpu.h"
#include "dali/operators/reader/reader_op.h"

namespace dali {

class FitsReaderGPU : public FitsReader<GPUBackend, FitsFileWrapperGPU> {
 public:
  explicit FitsReaderGPU(const OpSpec& spec) : FitsReader<GPUBackend, FitsFileWrapperGPU>(spec) {
    bool shuffle_after_epoch = spec.GetArgument<bool>("shuffle_after_epoch");
    loader_ = InitLoader<FitsLoaderGPU>(spec, shuffle_after_epoch);
  }

 protected:
  void RunImpl(Workspace& ws) override;
  using Operator<GPUBackend>::RunImpl;

 private:
  USE_READER_OPERATOR_MEMBERS(GPUBackend, FitsFileWrapperGPU, FitsFileWrapperGPU, true);
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_FITS_READER_GPU_OP_H_
