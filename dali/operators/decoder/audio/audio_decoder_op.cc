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

* output[0]: batch of decoded data
* output[1]: batch of sampling rates [Hz]

Sample rate (output[1]) at index `i` corresponds to sample (output[0]) at index `i`.
On the event more metadata will appear, we reserve a right to change this behaviour.)code")
                .NumInput(1)
                .NumOutput(detail::kNumOutputs)
.
AddOptionalArg(detail::kSampleRateName,
"If specified, the output data will be resampled to the frequency provided [Hz]",
-1)
                .AddOptionalArg(
                        detail::kOutputTypeName,
                        "Type of the output data. Supports types: `INT16`, `INT32`, `FLOAT`",
                        DALI_INT16);

DALI_REGISTER_OPERATOR(AudioDecoder, AudioDecoderCpu, CPU);


namespace {
double Hann(double x) {
  return 0.5 * (1 + std::cos(x * M_PI_2));
}


double sinc(double x) {
  x *= M_PI;
  if (std::abs(x) < 1e-10)
    return 1 - x * x * 0.25;  // approximate by a parabola near the pole
  return std::sin(x) / x;
}


struct sinc_coeffs {
  void init(int coeffs, int lobes = 3, std::function<double(double)> envelope = Hann) {
    float scale = 2.0f * lobes / (coeffs - 1);
    float scale_envelope = 2.0f / coeffs;
    this->coeffs = coeffs;
    this->lobes = lobes;
    lookup.resize(coeffs + 2); // add zeros
    center = (coeffs - 1) * 0.5f;
    for (int i = 0; i < coeffs; i++) {
      float x = (i - center) * scale;
      float y = (i - center) * scale_envelope;
      float w = sinc(x) * envelope(y);
      lookup[i + 1] = w;
      std::cerr << i << ": " << w << "\n";
    }
    center++;  // allow for leading zero
    this->scale = 1 / scale;
  }


  std::pair<int, int> input_range(float x) const {
    int i0 = std::ceil(x) - lobes;
    int i1 = std::floor(x) + lobes;
    return {i0, i1};
  }


  float operator()(float x) const {
    float fi = x * scale + center;
    int i = std::floor(fi);
    float di = fi - i;
    assert(i >= 0 && i < (int) lookup.size());
    return lookup[i] + di * (lookup[i + 1] - lookup[i]);
  }


  float scale = 1, center = 1;
  int lobes = 0, coeffs = 0;
  std::vector<float> lookup;
};


void resample_sinc(
        float *out, int64_t n_out, double out_rate,
        const float *in, int64_t n_in, double in_rate,
        int lobes = 3) {
  sinc_coeffs coeffs;
  coeffs.init(1024, lobes);
  int64_t in_pos = 0;
  int64_t block = 1 << 10;  // still leaves 13 significant bits for fractional part
  double scale = in_rate / out_rate;
  float fscale = scale;
  for (int64_t out_block = 0; out_block < n_out; out_block += block) {
    int64_t block_end = std::min(out_block + block, n_out);
    double in_block_f = (out_block + 0.5) * scale - 0.5;
    int64_t in_block_i = std::floor(in_block_f);
    float in_pos = in_block_f - in_block_i;
    const float *in_block_ptr = in + in_block_i;
    for (int64_t out_pos = out_block; out_pos < block_end; out_pos++, in_pos += fscale) {
      int i0 = std::ceil(in_pos) - lobes;
      int i1 = std::floor(in_pos) + lobes;
      if (i0 + in_block_i < 0)
        i0 = -in_block_i;
      if (i1 + in_block_i >= n_in)
        i1 = n_in - 1 - in_block_i;
      float f = 0;
      float x = i0 - in_pos;
      for (int i = i0; i <= i1; i++, x++) {
        assert(in_block_ptr + i >= in && in_block_ptr + i < in + n_in);
        float w = coeffs(x);
        f += in_block_ptr[i] * w;
      }
      assert(out_pos >= 0 && out_pos < n_out);
      out[out_pos] = f;
    }
  }
}


}  // namespace


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
//    TensorShape<> data_sample_shape = target_sample_rate_ == -1 ?
//                                      {meta.length, meta.channels} :
//                                      {static_cast<int>(meta.length * target_sample_rate /
//                                                        meta.sample_rate) + 1, 1};
    TensorShape<> data_sample_shape = {meta.length, meta.channels};
    shape_data.set_tensor_shape(i, data_sample_shape);
    shape_rate.set_tensor_shape(i, {1});
    files_names_[i] = input[i].GetSourceInfo();

    if (NeedsResampling()) {
      intermediate_buffers_[i].resize(meta.length);
    }

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
      auto &decoder = decoders_[i];
      auto &output = decoded_output[i];

      try {
        decoder->Decode({reinterpret_cast<char *>(output.raw_mutable_data()),
                         static_cast<int>(output.type().size() * output.shape().num_elements())});
        auto sample_rate_ptr = sample_rate_output[i].mutable_data<sample_rate_t>();
        *sample_rate_ptr =
                target_sample_rate_ == -1 ? sample_meta_[i].sample_rate : target_sample_rate_;
      } catch (const DALIException &e) {
        DALI_FAIL(make_string("Error decoding file.\nError: ", e.what(), "\nFile: ",
                              files_names_[i], "\n"));
      }

      if (NeedsResampling()) {
        if (sample_meta_[i].channels != 1) {
//          detail::Downmixing()
        }
      }
    };

    tp.DoWorkWithID(decode);
  }
  tp.WaitForWork();
}


}  // namespace dali
