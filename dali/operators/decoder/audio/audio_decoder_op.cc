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
#include "dali/operators/decoder/audio/audio_decoder_impl.h"
#include "dali/pipeline/operator/op_schema.h"
#include "dali/pipeline/data/views.h"

namespace dali {

DALI_REGISTER_OPERATOR(decoders__Audio, AudioDecoderCpu, CPU);

DALI_SCHEMA(decoders__Audio)
  .DocStr(R"code(Decodes waveforms from encoded audio data.

It supports the following audio formats: wav, flac and ogg.
This operator produces the following outputs:

* output[0]: A batch of decoded data
* output[1]: A batch of sampling rates [Hz].
)code")
  .NumInput(1)
  .NumOutput(2)
  .AddOptionalArg("downmix", R"code(If set to True, downmix all input channels to mono.

If downmixing is turned on, the decoder output is 1D.
If downmixing is turned off, it produces 2D output with interleaved channels.)code", false)
  .AddOptionalArg("dtype", R"code(Output data type.

Supported types: ``INT16``, ``INT32``, ``FLOAT``.)code", DALI_FLOAT)
  .AddOptionalArg("sample_rate",
          "If specified, the target sample rate, in Hz, to which the audio is resampled.",
          0.0f, true)
  .AddOptionalArg("quality", R"code(Resampling quality, where 0 is the lowest, and 100 is
the highest.

0 gives 3 lobes of the sinc filter, 50 gives 16 lobes, and 100 gives 64 lobes.)code",
          50.0f, false);


DALI_REGISTER_OPERATOR(AudioDecoder, AudioDecoderCpu, CPU);

DALI_SCHEMA(AudioDecoder)
    .DocStr("Legacy alias for :meth:`decoders.audio`.")
    .NumInput(1)
    .NumOutput(2)
    .AddParent("decoders__Audio")
    .MakeDocPartiallyHidden()
    .Deprecate(
        "decoders__Audio",
        R"code(In DALI 1.0 all decoders were moved into a dedicated :mod:`~nvidia.dali.fn.decoders`
submodule and renamed to follow a common pattern. This is a placeholder operator with identical
functionality to allow for backward compatibility.)code");  // Deprecated in 1.0;

bool
AudioDecoderCpu::SetupImpl(std::vector<OutputDesc> &output_desc, const workspace_t<Backend> &ws) {
  auto &input = ws.template InputRef<Backend>(0);
  const auto batch_size = input.shape().num_samples();
  GetPerSampleArgument<float>(target_sample_rates_, "sample_rate", ws, batch_size);

  for (int i = 0; i < batch_size; i++) {
    DALI_ENFORCE(input.shape()[i].size() == 1, "Raw input must be 1D encoded byte data");
  }
  DALI_ENFORCE(IsType<uint8_t>(input.type()), "Raw files must be stored as uint8 data.");
  decoders_.resize(batch_size);
  sample_meta_.resize(batch_size);
  files_names_.resize(batch_size);

  decode_type_ = use_resampling_ ? DALI_FLOAT : output_type_;
  for (int i = 0; i < batch_size; i++)
    decoders_[i] = make_generic_audio_decoder();

  output_desc.resize(2);

  TensorListShape<> shape_rate(batch_size, 0);
  TensorListShape<> shape_data(batch_size, downmix_ ? 1 : 2);

  for (int i = 0; i < batch_size; i++) {
    auto &meta = sample_meta_[i] =
        decoders_[i]->Open({reinterpret_cast<const char *>(input[i].raw_mutable_data()),
                            input[i].shape().num_elements()});
    TensorShape<> data_sample_shape = DecodedAudioShape(
        meta, use_resampling_ ? target_sample_rates_[i] : -1.0f, downmix_);
    shape_data.set_tensor_shape(i, data_sample_shape);
    shape_rate.set_tensor_shape(i, {});
    files_names_[i] = input[i].GetSourceInfo();
  }

  output_desc[0] = { shape_data, TypeTable::GetTypeInfo(output_type_) };
  output_desc[1] = { shape_rate, TypeTable::GetTypeInfo(DALI_FLOAT) };
  return true;
}

template<typename OutputType>
void
AudioDecoderCpu::DecodeSample(const TensorView<StorageCPU, OutputType, DynamicDimensions> &audio,
                              int thread_idx, int sample_idx) {
  auto &meta = sample_meta_[sample_idx];
  float target_sr = use_resampling_ ? target_sample_rates_[sample_idx] : meta.sample_rate;
  bool should_resample = target_sr != meta.sample_rate;
  bool should_downmix = meta.channels > 1 && downmix_;
  int64_t decode_scratch_sz = 0;
  int64_t resample_scratch_sz = 0;
  if (should_resample || should_downmix)
    decode_scratch_sz = meta.length * meta.channels;

  // resample scratch is used to prepare a single or multiple (depending if
  // downmixing is needed) channel float input, required by the resampling
  // kernel
  if (should_resample)
    resample_scratch_sz = should_downmix ? meta.length : meta.length * meta.channels;

  auto &scratch_decoder = scratch_decoder_[thread_idx];
  scratch_decoder.resize(decode_scratch_sz);

  auto &scratch_resampler = scratch_resampler_[thread_idx];
  scratch_resampler.resize(resample_scratch_sz);

  // TODO(janton): handle offset and duration

  DecodeAudio<OutputType>(
    audio, *decoders_[sample_idx], meta, resampler_,
    {scratch_decoder.data(), decode_scratch_sz},
    {scratch_resampler.data(), resample_scratch_sz},
    target_sr, downmix_,
    files_names_[sample_idx].c_str());
}

template <typename OutputType>
void AudioDecoderCpu::DecodeBatch(workspace_t<Backend> &ws) {
  auto decoded_output = view<OutputType, DynamicDimensions>(ws.template OutputRef<Backend>(0));
  auto sample_rate_output = view<float, 0>(ws.template OutputRef<Backend>(1));
  int batch_size = decoded_output.shape.num_samples();
  auto &tp = ws.GetThreadPool();

  scratch_decoder_.resize(tp.NumThreads());
  scratch_resampler_.resize(tp.NumThreads());

  for (int i = 0; i < batch_size; i++) {
    tp.AddWork([&, i](int thread_id) {
      try {
        DecodeSample<OutputType>(decoded_output[i], thread_id, i);
        sample_rate_output[i].data[0] = use_resampling_
          ? target_sample_rates_[i]
          : sample_meta_[i].sample_rate;
      } catch (const DALIException &e) {
        DALI_FAIL(make_string("Error decoding file ", files_names_[i], ". Error: ", e.what()));
      }
    }, sample_meta_[i].length * sample_meta_[i].channels);
  }

  tp.RunAll();
}


void AudioDecoderCpu::RunImpl(workspace_t<Backend> &ws) {
  TYPE_SWITCH(output_type_, type2id, OutputType, (int16_t, int32_t, float), (
    DecodeBatch<OutputType>(ws);
  ), DALI_FAIL(make_string("Unsupported output type: ", output_type_)))  // NOLINT
}

}  // namespace dali
