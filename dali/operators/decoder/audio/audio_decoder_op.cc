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
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_SCHEMA(AudioDecoder)
  .DocStr(R"code(Decode audio data.
This operator is a generic way of handling encoded data in DALI.
It supports most of well-known audio formats (wav, flac, ogg).

This operator produces two outputs:

* output[0]: batch of decoded data
* output[1]: batch of sampling rates [Hz]
)code")
  .NumInput(1)
  .NumOutput(2)
  .AddOptionalArg("sample_rate",
          "If specified, the target sample rate, in Hz, to which the audio is resampled.",
          0.0f, true)
  .AddOptionalArg("quality",
          "Resampling quality, 0 is lowest, 100 is highest.\n"
          "0 corresponds to 3 lobes of the sinc filter; "
          "50 gives 16 lobes and 100 gives 64 lobes.",
          50.0f, false)
  .AddOptionalArg("downmix",
          "If True, downmix all input channels to mono. "
          "If downmixing is turned on, decoder will produce always 1-D output", false)
  .AddOptionalArg("dtype",
          "Type of the output data. Supports types: `INT16`, `INT32`, `FLOAT`", DALI_FLOAT);

DALI_REGISTER_OPERATOR(AudioDecoder, AudioDecoderCpu, CPU);

bool
AudioDecoderCpu::SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) {
  GetPerSampleArgument<float>(target_sample_rates_, "sample_rate", ws);
  auto &input = ws.template InputRef<Backend>(0);
  const auto batch_size = input.shape().num_samples();

  for (int i = 0; i < batch_size; i++) {
    DALI_ENFORCE(input.shape()[i].size() == 1, "Raw input must be 1D encoded byte data");
  }
  DALI_ENFORCE(IsType<uint8_t>(input.type()), "Raw files must be stored as uint8 data.");
  decoders_.resize(batch_size);
  intermediate_buffers_.resize(batch_size);
  sample_meta_.resize(batch_size);
  files_names_.resize(batch_size);

  decode_type_ = use_resampling_ || downmix_ ? DALI_FLOAT : output_type_;
  TYPE_SWITCH(decode_type_, type2id, OutputType, (int16_t, int32_t, float), (
      for (int i=0; i < batch_size; i++)
        decoders_[i] = std::make_unique<GenericAudioDecoder<OutputType>>();
  ), DALI_FAIL("Unsupported output type"))  // NOLINT

  output_desc.resize(2);

  // Currently, metadata is only the sampling rate.
  // On the event something else would emerge,
  // this approach should be completely redefined
  TensorListShape<> shape_rate(batch_size, 1);
  TensorListShape<> shape_data(batch_size, downmix_ ? 1 : 2);

  for (int i = 0; i < batch_size; i++) {
    auto meta = decoders_[i]->Open({reinterpret_cast<const char *>(input[i].raw_mutable_data()),
                                    input[i].shape().num_elements()});
    sample_meta_[i] = meta;
    int64_t out_length = OutputLength(meta.length, meta.sample_rate, i);
    TensorShape<> data_sample_shape;
    if (downmix_) {
      data_sample_shape = {out_length};
    } else {
      data_sample_shape = {out_length, meta.channels};
    }

    shape_data.set_tensor_shape(i, data_sample_shape);
    shape_rate.set_tensor_shape(i, {1});
    files_names_[i] = input[i].GetSourceInfo();
  }

  output_desc[0] = { shape_data, TypeTable::GetTypeInfo(output_type_) };
  output_desc[1] = { shape_rate, TypeTable::GetTypeInfo(DALI_FLOAT) };
  return true;
}


template <typename T>
span<char> as_raw_span(T *buffer, ptrdiff_t length) {
  return make_span(reinterpret_cast<char*>(buffer), length*sizeof(T));
}


template<typename OutputType>
void
AudioDecoderCpu::DecodeSample(const TensorView<StorageCPU, OutputType, DynamicDimensions> &audio,
                              int thread_idx, int sample_idx) {
  const AudioMetadata &meta = sample_meta_[sample_idx];

  auto &tmp_buf = intermediate_buffers_[thread_idx];
  double output_rate = meta.sample_rate;
  if (use_resampling_) {
    output_rate = target_sample_rates_[sample_idx];
    DALI_ENFORCE(meta.sample_rate > 0, make_string("Unknown or invalid input sampling rate."));
    DALI_ENFORCE(output_rate > 0, make_string(
        "Output sampling rate must be positive; got ", output_rate));
  }
  bool should_resample = meta.sample_rate != output_rate;
  bool should_downmix = meta.channels > 1 && downmix_;
  if (should_resample || should_downmix || output_type_ != decode_type_) {
    assert(decode_type_ == DALI_FLOAT);
    int64_t tmp_size = should_downmix && should_resample
      ? meta.length * (meta.channels + 1)   // downmix to intermediate buffer, then resample
      : meta.length * meta.channels;        // decode to intermediate, then resample or downmix
                                            // directly to the output

    tmp_buf.resize(tmp_size);
    decoders_[sample_idx]->Decode(as_raw_span(tmp_buf.data(), meta.length * meta.channels));

    if (should_downmix) {
      if (should_resample) {
        // downmix and resample
        float *downmixed = tmp_buf.data() + meta.length * meta.channels;
        assert(downmixed + meta.length <= tmp_buf.data() + tmp_buf.size());
        kernels::signal::Downmix(downmixed, tmp_buf.data(), meta.length, meta.channels);
        resampler_.Resample(audio.data, 0, audio.shape[0], output_rate,
                            downmixed, meta.length, meta.sample_rate);
      } else {
        // downmix only
        kernels::signal::Downmix(audio.data, tmp_buf.data(), meta.length, meta.channels);
      }
    } else if (should_resample) {
      // multi-channel resample
      resampler_.Resample(audio.data, 0, audio.shape[0], output_rate,
                          tmp_buf.data(), meta.length, meta.sample_rate, meta.channels);

    } else {
      // convert or copy only - this will only happen if resampling is specified, but this
      // recording's sampling rate and number of channels coincides with the target
      int64_t len = std::min<int64_t>(volume(audio.shape), meta.length*meta.channels);
      for (int64_t ofs = 0; ofs < len; ofs++) {
        audio.data[ofs] = ConvertSatNorm<OutputType>(tmp_buf[ofs]);
      }
    }
  } else {
    assert(!should_downmix && !should_resample);
    decoders_[sample_idx]->Decode(as_raw_span(audio.data, volume(audio.shape)));
  }
}

template <typename OutputType>
void AudioDecoderCpu::DecodeBatch(workspace_t<Backend> &ws) {
  auto decoded_output = view<OutputType, DynamicDimensions>(ws.template OutputRef<Backend>(0));
  auto sample_rate_output = view<float, 1>(ws.template OutputRef<Backend>(1));
  int batch_size = decoded_output.shape.num_samples();
  auto &tp = ws.GetThreadPool();

  intermediate_buffers_.resize(tp.size());

  for (int i = 0; i < batch_size; i++) {
    tp.DoWorkWithID([&, i](int thread_id) {
      try {
        DecodeSample<OutputType>(decoded_output[i], thread_id, i);
        sample_rate_output[i].data[0] = use_resampling_
          ? target_sample_rates_[i]
          : sample_meta_[i].sample_rate;
      } catch (const DALIException &e) {
        DALI_FAIL(make_string("Error decoding file.\nError: ", e.what(), "\nFile: ",
                              files_names_[i], "\n"));
      }
    });
  }

  tp.WaitForWork();
}


void AudioDecoderCpu::RunImpl(workspace_t<Backend> &ws) {
  TYPE_SWITCH(output_type_, type2id, OutputType, (int16_t, int32_t, float), (
      DecodeBatch<OutputType>(ws);
  ), DALI_FAIL("Unsupported output type"))  // NOLINT
}

}  // namespace dali
