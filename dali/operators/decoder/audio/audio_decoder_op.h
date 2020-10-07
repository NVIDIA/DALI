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
#include "dali/kernels/signal/resampling.h"
#include "dali/kernels/signal/downmixing.h"
#include "dali/core/tensor_view.h"

namespace dali {

class AudioDecoderCpu : public Operator<CPUBackend> {
 private:
  using Backend = CPUBackend;

 public:
  explicit inline AudioDecoderCpu(const OpSpec &spec) :
          Operator<Backend>(spec),
          output_type_(spec.GetArgument<DALIDataType>("dtype")),
          downmix_(spec.GetArgument<bool>("downmix")),
          use_resampling_(spec.HasArgument("sample_rate") || spec.HasTensorArgument("sample_rate")),
          quality_(spec.GetArgument<float>("quality")) {
    if (use_resampling_) {
      double q = quality_;
      DALI_ENFORCE(q >= 0 && q <= 100, "Resampling quality must be in [0..100] range");
      // this should give 3 lobes for q = 0, 16 lobes for q = 50 and 64 lobes for q = 100
      int lobes = std::round(0.007 * q * q - 0.09 * q + 3);
      resampler_.Initialize(lobes, lobes * 64 + 1);
    }
  }

  inline ~AudioDecoderCpu() override = default;

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override;

  void RunImpl(workspace_t<Backend> &ws) override;


  bool CanInferOutputs() const override {
    return true;
  }


 private:
  template<typename OutputType>
  void DecodeSample(const TensorView<StorageCPU, OutputType, DynamicDimensions> &audio,
                    int thread_idx, int sample_idx);

  template <typename OutputType>
  void DecodeBatch(workspace_t<Backend> &ws);

  int64_t OutputLength(int64_t in_length, double in_rate, int sample_idx) const {
    if (use_resampling_) {
      return kernels::signal::resampling::resampled_length(
          in_length, in_rate, target_sample_rates_[sample_idx]);
    } else {
      return in_length;
    }
  }

  std::vector<float> target_sample_rates_;
  kernels::signal::resampling::Resampler resampler_;
  DALIDataType output_type_ = DALI_NO_TYPE, decode_type_ = DALI_NO_TYPE;
  const bool downmix_ = false, use_resampling_ = false;
  const float quality_ = 50.0f;
  std::vector<std::string> files_names_;
  std::vector<AudioMetadata> sample_meta_;
  std::vector<vector<float>> scratch_decoder_;
  std::vector<vector<float>> scratch_resampler_;
  std::vector<std::unique_ptr<AudioDecoderBase>> decoders_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_AUDIO_AUDIO_DECODER_OP_H_
