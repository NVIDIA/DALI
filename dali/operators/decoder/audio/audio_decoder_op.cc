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
#include "dali/operators/decoder/audio/utils.h"
#include "dali/pipeline/operator/op_schema.h"

namespace dali {

DALI_SCHEMA(AudioDecoder)
                .DocStr(R"code(Decode audio data.
This operator is a generic way of handling encoded data in DALI.
It supports most of well-known audio formats (wav, flac, ogg).

This operator produces two outputs: <br/>
output[0]: batch of decoded data <br/>
output[1]: batch of sampling rates [Hz]

Sample rate (output[1]) at index `i` corresponds to sample (output[0]) at index `i`.
On the event more metadata will appear, we reserve a right to change this behaviour.)code")
                .NumInput(1)
                .NumOutput(detail::kNumOutputs)
                .AddOptionalArg(detail::kSampleRateName,
                                "If specified, the output data will be resampled "
                                "to the frequency provided [Hz]", -1)
                .AddOptionalArg(
                        detail::kOutputTypeName,
                        "Type of the output data. Supports types: `INT16`, `INT32`, `FLOAT`",
                        DALI_INT16);

DALI_REGISTER_OPERATOR(AudioDecoder, AudioDecoderCpu, CPU);

std::vector<std::string> AudioDecoderCpu::files_names_ = {};
std::vector<AudioMetadata> AudioDecoderCpu::sample_meta_ = {};
std::vector<std::vector<float>> AudioDecoderCpu::intermediate_buffers_ = {};

class AudioDecoderCpu::DecoderHelper {
 public:
  explicit DecoderHelper(AudioDecoderBase *decoder) : decoder_(decoder) {}


  virtual void operator()(span<char> output_buffer, int &sample_rate, int sample_idx) = 0;

 protected:
  AudioDecoderBase *decoder_;
};

class AudioDecoderCpu::DirectDecoder : public AudioDecoderCpu::DecoderHelper {
 public:
  explicit DirectDecoder(AudioDecoderBase *decoder) : DecoderHelper(decoder) {}


  void operator()(span<char> output_buffer, int &sample_rate, int sample_idx) override {
    decoder_->Decode(output_buffer);
    sample_rate = sample_meta_[sample_idx].sample_rate;
  }


 private:
  using DecoderHelper::decoder_;
};

using T = float;

class AudioDecoderCpu::ResamplingDecoder : public AudioDecoderCpu::DecoderHelper {
 public:
  ResamplingDecoder(AudioDecoderBase *decoder, int target_sample_rate) :
          DecoderHelper(decoder), target_sample_rate_(target_sample_rate) {}


  void operator()(span<char> output_buffer, int &sample_rate, int sample_idx) override {
    intermediate_buffers_[sample_idx].resize(sample_meta_[sample_idx].length);
    auto intermediate_buffer = make_span(intermediate_buffers_[sample_idx]);
    auto intermediate_buffer_c = make_cspan(intermediate_buffers_[sample_idx]);
    auto typed = (TypedAudioDecoderBase<T> *) (decoder_);
    typed->DecodeTyped(intermediate_buffer);

    // You don't need to downmix mono-channel stream
    if (sample_meta_[sample_idx].channels > 1) {
      Downmixing(intermediate_buffer, intermediate_buffer_c, sample_meta_[sample_idx].channels);
    }

    resampling::resample_sinc(reinterpret_cast<float *>(output_buffer.data()),
                  output_buffer.size() / sizeof(float), target_sample_rate_,
                  intermediate_buffer.data(), intermediate_buffer.size(),
                  sample_meta_[sample_idx].sample_rate);
  }


 private:
  int target_sample_rate_;
};


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
  intermediate_buffers_.resize(batch_size);
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
    TensorShape<> data_sample_shape;
    if (!NeedsResampling()) {
      data_sample_shape = {meta.length, meta.channels};
    } else {
      data_sample_shape = {static_cast<int>(meta.length * target_sample_rate_ / meta.sample_rate) +
                           1, 1 /* resampling currently supported for 1-channel data only */};
    }
    shape_data.set_tensor_shape(i, data_sample_shape);
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
    auto decode = [&, i](int thread_id) {
        auto &output = decoded_output[i];
        DirectDecoder decoder(decoders_[i].get());
        try {
          decoder({reinterpret_cast<char *>(output.raw_mutable_data()),
                   static_cast<int>(output.type().size() * output.shape().num_elements())},
                  *sample_rate_output[i].mutable_data<sample_rate_t>(), i);
        } catch (const DALIException &e) {
          DALI_FAIL(make_string("Error decoding file.\nError: ", e.what(), "\nFile: ",
                                files_names_[i], "\n"));
        }
    };

    auto decode_and_resample = [&, i](int thread_id) {
        auto &output = decoded_output[i];
        ResamplingDecoder decoder(decoders_[i].get(), target_sample_rate_);
        try {
          decoder({reinterpret_cast<char *>(output.raw_mutable_data()),
                   static_cast<int>(output.type().size() * output.shape().num_elements())},
                  *sample_rate_output[i].mutable_data<sample_rate_t>(), i);
        } catch (const DALIException &e) {
          DALI_FAIL(make_string("Error decoding file.\nError: ", e.what(), "\nFile: ",
                                files_names_[i], "\n"));
        }
    };

    if (NeedsResampling()) {
      tp.DoWorkWithID(decode_and_resample);
    } else {
      tp.DoWorkWithID(decode);
    }
  }
  tp.WaitForWork();
}

}  // namespace dali
