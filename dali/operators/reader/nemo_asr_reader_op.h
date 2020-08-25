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
  explicit NemoAsrReader(const OpSpec& spec):
      DataReader<CPUBackend, AsrSample>(spec),
      read_sr_(spec.GetArgument<bool>("read_sample_rate")),
      read_text_(spec.GetArgument<bool>("read_text")) {
    loader_ = InitLoader<NemoAsrLoader>(spec);
  }

  void RunImpl(SampleWorkspace &ws) override {
    const AsrSample& sample = GetSample(ws.data_idx());

    auto &audio = ws.Output<CPUBackend>(0);
    audio.Copy(sample.audio, 0);

    int next_out_idx = 1;
    if (read_sr_) {
      auto &sample_rate = ws.Output<CPUBackend>(next_out_idx++);
      sample_rate.Resize({1});
      sample_rate.set_type(TypeTable::GetTypeInfo(DALI_UINT8));
      sample_rate.mutable_data<float>()[0] = sample.audio_meta.sample_rate;
      sample_rate.SetSourceInfo(sample.audio.GetSourceInfo());
    }

    if (read_text_) {
      auto &text = ws.Output<CPUBackend>(next_out_idx++);
      text.set_type(TypeTable::GetTypeInfo(DALI_UINT8));
      int64_t text_sz = sample.text.length() + 1;  // +1 for null character
      text.Resize({text_sz});
      std::memcpy(text.mutable_data<uint8_t>(), sample.text.c_str(), sample.text.length());
      text.mutable_data<uint8_t>()[sample.text.length()] = '\0';
      text.SetSourceInfo(sample.audio.GetSourceInfo());
    }
  }

 private:
  bool read_sr_;
  bool read_text_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_READER_NEMO_ASR_READER_OP_H_
