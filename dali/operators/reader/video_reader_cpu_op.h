// Copyright (c) 2017-2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef DALI_OPERATORS_READER_VIDEO_READER_CPU_OP_H_
#define DALI_OPERATORS_READER_VIDEO_READER_CPU_OP_H_

#include <string>
#include <vector>
#include <algorithm>

#include "dali/operators/reader/loader/video_loader.h"
#include "dali/operators/reader/reader_op.h"

namespace dali {

class VideoLoaderCPU : public Loader<CPUBackend, SequenceWrapper> {
public:
  explicit inline VideoLoaderCPU(const OpSpec &spec) : 
    Loader<CPUBackend, SequenceWrapper>(spec) {
  }

  void ReadSample(SequenceWrapper &tensor) override {
  }

protected:
  Index SizeImpl() override {
    return 0;
  }

private:
  void Reset(bool wrap_to_shard) override {
  } 

};


class VideoReaderCPU : public DataReader<CPUBackend, SequenceWrapper> {
 public:
  explicit VideoReaderCPU(const OpSpec &spec)
      : DataReader<CPUBackend, SequenceWrapper>(spec) {
        loader_ = InitLoader<VideoLoaderCPU>(spec);
      }
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_VIDEO_READER_CPU_OP_H_
