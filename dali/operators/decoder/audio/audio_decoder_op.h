


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

const std::string kOutputTypeName = "dtype";  // NOLINT
const int kNumOutputs = 2;

}  // namespace detail

class AudioDecoderCpu : public Operator<CPUBackend> {
 private:
  using Backend = CPUBackend;

 public:
  explicit inline AudioDecoderCpu(const OpSpec &spec) :
          Operator<Backend>(spec),
          output_type_(spec.GetArgument<DALIDataType>(detail::kOutputTypeName)) {}


  inline ~AudioDecoderCpu() override = default;

 protected:
  bool SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) override {
    auto &input = ws.template InputRef<Backend>(0);

    for (int i = 0; i < input.shape().num_samples(); i++) {
      DALI_ENFORCE(input.shape()[i].size() == 1, "Input must be 1D encoded byte data");
    }
    DALI_ENFORCE(IsType<uint8_t>(input.type()), "Input must be stored as uint8_t data.");
    TypeInfo type;
    TypeInfo type_i32;
    type_i32.SetType<int>(DALI_INT32);

    TYPE_SWITCH(output_type_, type2id, OutputType, (int16_t, int32_t, float), (
            for (int i=0; i < input.shape().num_samples(); i++) {
            decoders_.emplace_back(new GenericAudioDecoder<OutputType>());
    }
            type.SetType<OutputType>(output_type_);
    ), DALI_FAIL("Unsupported output type"))  // NOLINT

    output_desc.resize(detail::kNumOutputs);

    // Currently, metadata is only the sampling rate.
    // On the event something else would emerge,
    // this approach should be completely redefined
    TensorListShape<> shape_rate(input.shape().num_samples(), 1);
    TensorListShape<> shape_data(input.shape().num_samples(), 2);

    for (int i = 0; i < input.shape().num_samples(); i++) {
      auto meta = decoders_[i]->Open(
              {(const char *) input[i].raw_mutable_data(), input[i].shape().num_elements()});
      samples_meta_.emplace_back(meta);
      shape_data.set_tensor_shape(i, {meta.channels, meta.length});
      shape_rate.set_tensor_shape(i, {1});
      files_names_.emplace_back(input[i].GetSourceInfo());
    }
    output_desc[0] = {shape_data, type};
    output_desc[1] = {shape_rate, type_i32};
    return true;
  }


  void RunImpl(workspace_t<Backend> &ws) override {
    auto &decoded_output = ws.template OutputRef<Backend>(0);
    auto &sample_rate_output = ws.template OutputRef<Backend>(1);

    for (int i = 0; i < decoded_output.shape().num_samples(); i++) {
      try {
        decoders_[i]->Decode({reinterpret_cast<char *>(decoded_output[i].raw_mutable_data()),
                              static_cast<int>(decoded_output[i].type().size() *
                                               decoded_output[i].shape().num_elements())});
        auto sample_rate_ptr =
                reinterpret_cast<sample_rate_t *>(sample_rate_output[i].raw_mutable_data());
        *sample_rate_ptr = samples_meta_[i].sample_rate;
      } catch (const DALIException &e) {
        DALI_FAIL(make_string("Error decoding file.\nError: ", e.what(),
                              "\nFile: ", files_names_[i], "\n"));
      }
    }
  }


  bool CanInferOutputs() const override {
    return true;
  }


 private:
  DALIDataType output_type_;
  std::vector<std::string> files_names_;
  std::vector<AudioMetadata> samples_meta_;
  using sample_rate_t = decltype(decltype(samples_meta_)::value_type::sample_rate);
  std::vector<std::unique_ptr<AudioDecoderBase>> decoders_;
};

}  // namespace dali

#endif  // DALI_OPERATORS_DECODER_AUDIO_AUDIO_DECODER_OP_H_
