


// Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

#ifndef DALI_OPERATORS_DECODER_AUDIO_AUDIO_DECODER_OP_H_
#define DALI_OPERATORS_DECODER_AUDIO_AUDIO_DECODER_OP_H_

#include <memory>
#include <string>
#include <vector>
#include "dali/core/static_switch.h"
#include "dali/operators/decoder/audio/audio_decoder.h"
#include "dali/operators/decoder/audio/generic_decoder.h"
#include "dali/pipeline/data/backend.h"
#include "dali/pipeline/workspace/workspace.h"
#include "dali/pipeline/operator/operator.h"
#include "dali/pipeline/workspace/host_workspace.h"

namespace dali {

namespace detail {

const std::string kOutputTypeName = "dtype";        // NOLINT
const std::string kSampleRateName = "sample_rate";  // NOLINT
const int kNumOutputs = 2;


/**
 * Downmixing data to mono-channel. It's safe to do it in-place.
 */
template<typename T>
void Downmixing(T *out, const T *in, size_t n_in, size_t n_channels_in) {
  DALI_ENFORCE(n_channels_in > 1, "You can't downmix mono-channel data");
  DALI_ENFORCE(n_in % n_channels_in == 0, "Data layout incorrect");
  for (size_t i = 0; i < n_in; i++) {
    *out = *in++;
    for (size_t j = 1; j < n_channels_in; j++) {
      *out += *in++;
    }
    *out++ /= n_channels_in;
  }
}

}  // namespace detail

class AudioDecoderCpu : public Operator<CPUBackend> {
 private:
  using Backend = CPUBackend;

 public:
  explicit inline AudioDecoderCpu(const OpSpec &spec) :
          Operator<Backend>(spec),
          output_type_(spec.GetArgument<DALIDataType>(detail::kOutputTypeName)),
          target_sample_rate_(spec.GetArgument<int>(detail::kSampleRateName)) {
    DALI_ENFORCE(target_sample_rate_ != -1 && output_type_ == DALI_FLOAT,
                 "Resampling currently supported only for float output");
  }

  inline ~AudioDecoderCpu() override = default;

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override;

  void RunImpl(workspace_t<Backend> &ws) override;

  bool CanInferOutputs() const override {
    return true;
  }

 private:


  bool NeedsResampling() {
    return target_sample_rate_ != -1;
  }


  DALIDataType output_type_;
  const int target_sample_rate_;
  std::vector<std::string> files_names_;
  std::vector<AudioMetadata> sample_meta_;
  using sample_rate_t = decltype(AudioMetadata::sample_rate);
  std::vector<std::unique_ptr<AudioDecoderBase>> decoders_;
  std::vector <std::vector<float>> intermediate_buffers_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_AUDIO_AUDIO_DECODER_OP_H_
