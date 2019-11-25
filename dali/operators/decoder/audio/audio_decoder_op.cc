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

#include "dali/operators/decoder/audio/audio_decoder_op.h"
#include "dali/pipeline/operator/op_schema.h"

namespace dali {

DALI_SCHEMA(AudioDecoder)
                .DocStr(R"code(Decode audio data.
This operator is a generic way of handling encoded data in DALI.
It supports most of well-known audio formats (wav, flac, ogg).

This operator produces two outputs:
output[0]: batch of decoded data
output[1]: batch of sampling rates [Hz]

Sample rate (output[1]) at index `i` corresponds to sample (output[0]) at index `i`.
On the event more metadata will appear, we reserve a right to change this behaviour.)code")
                .NumInput(1)
                .NumOutput(detail::kNumOutputs)
                .AddOptionalArg(
                        detail::kOutputTypeName,
                        "Type of the output data. Supports types: `INT16`, `INT32`, `FLOAT`",
                        DALI_INT16);

DALI_REGISTER_OPERATOR(AudioDecoder, AudioDecoderCpu, CPU);


bool
AudioDecoderCpu::SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) {
  auto &input = ws.template InputRef<Backend>(0);
  const auto batch_size = input.shape().num_samples();

  for (int i = 0; i < batch_size; i++) {
    DALI_ENFORCE(input.shape()[i].size() == 1, "Input must be 1D encoded byte data");
  }
  DALI_ENFORCE(IsType<uint8_t>(input.type()), "Input must be stored as uint8_t data.");
  TypeInfo type;
  TypeInfo type_i32;
  type_i32.SetType<int>(DALI_INT32);
  decoders_.resize(batch_size);
  sample_meta_.resize(batch_size);
  files_names_.resize(batch_size);

  TYPE_SWITCH(output_type_, type2id, OutputType, (int16_t, int32_t, float), (
          for (int i=0; i < batch_size; i++) {
            decoders_[i] = std::make_unique<GenericAudioDecoder<OutputType>>();
          }
          type.SetType<OutputType>(output_type_);
  ), DALI_FAIL("Unsupported output type"))  // NOLINT

  output_desc.resize(detail::kNumOutputs);

  // Currently, metadata is only the sampling rate.
  // On the event something else would emerge,
  // this approach should be completely redefined
  TensorListShape<> shape_rate(batch_size, 1);
  TensorListShape<> shape_data(batch_size, 2);

  for (int i = 0; i < batch_size; i++) {
    auto meta = decoders_[i]->Open({reinterpret_cast<const char *>(input[i].raw_mutable_data()),
                                    input[i].shape().num_elements()});
    sample_meta_[i] = meta;
    shape_data.set_tensor_shape(i, {meta.length, meta.channels});
    shape_rate.set_tensor_shape(i, {1});
    files_names_[i] = input[i].GetSourceInfo();
  }
  output_desc[0] = {shape_data, type};
  output_desc[1] = {shape_rate, type_i32};
  return true;
}


void AudioDecoderCpu::RunImpl(workspace_t<Backend> &ws) {
  auto &decoded_output = ws.template OutputRef<Backend>(0);
  auto &sample_rate_output = ws.template OutputRef<Backend>(1);
  auto &tp = ws.GetThreadPool();
  auto batch_size = decoded_output.shape().num_samples();

  for (int i = 0; i < batch_size; i++) {
    tp.DoWorkWithID([&, i](int thread_id) {
        auto &decoder = decoders_[i];
        auto &output = decoded_output[i];
        try {
          decoder->Decode({reinterpret_cast<char *>(output.raw_mutable_data()),
                           static_cast<int>(output.type().size() * output.shape().num_elements())});
          auto sample_rate_ptr = sample_rate_output[i].mutable_data<sample_rate_t>();
          *sample_rate_ptr = sample_meta_[i].sample_rate;
        } catch (const DALIException &e) {
          DALI_FAIL(make_string("Error decoding file.\nError: ", e.what(), "\nFile: ",
                                files_names_[i], "\n"));
        }
    });
  }
}

}  // namespace dali
