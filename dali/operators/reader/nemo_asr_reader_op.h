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

#ifndef DALI_OPERATORS_READER_NEMO_ASR_READER_OP_H_
#define DALI_OPERATORS_READER_NEMO_ASR_READER_OP_H_

#include <fstream>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include <istream>
#include <memory>

#include "dali/operators/reader/reader_op.h"
#include "dali/operators/reader/loader/nemo_asr_loader.h"

namespace dali {
class NemoAsrReader : public DataReader<CPUBackend, AsrSample> {
 public:
  explicit NemoAsrReader(const OpSpec& spec);
  ~NemoAsrReader() override;

 protected:
  void Prefetch() override;
  void RunImpl(SampleWorkspace &ws) override;

 private:
  Tensor<CPUBackend>& GetDecodedAudioSample(int sample_idx);

  bool read_sr_;
  bool read_text_;
  bool read_idxs_;
  DALIDataType dtype_;

  int num_threads_;
  ThreadPool thread_pool_;

  // prefetch_depth * batch_size set of buffers that we reuse to decode audio
  using TensorVectorPtr = std::unique_ptr<TensorVector<CPUBackend>>;
  std::vector<TensorVectorPtr> prefetched_decoded_audio_;

  std::unordered_map<void*, int> decoded_map_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_NEMO_ASR_READER_OP_H_
