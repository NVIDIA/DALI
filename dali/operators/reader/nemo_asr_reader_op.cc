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

#include "dali/operators/reader/nemo_asr_reader_op.h"

namespace dali {

namespace {

int NemoAsrReaderOutputFn(const OpSpec &spec) {
  return static_cast<int>(spec.GetArgument<bool>("read_sample_rate")) +
         static_cast<int>(spec.GetArgument<bool>("read_text")) +
         static_cast<int>(spec.GetArgument<bool>("read_idxs"));
}

}  // namespace

DALI_REGISTER_OPERATOR(readers__NemoAsr, NemoAsrReader, CPU);

DALI_SCHEMA(readers__NemoAsr)
  .NumInput(0)
  .NumOutput(1)
  .DocStr(R"code(Reads automatic speech recognition (ASR) data (audio, text) from an
NVIDIA NeMo compatible manifest.

Example manifest file::

    {"audio_filepath": "path/to/audio1.wav", "duration": 3.45, "text": "this is a nemo tutorial"}
    {"audio_filepath": "path/to/audio1.wav", "offset": 3.45, "duration": 1.45, "text": "same audio file but using offset"}
    {"audio_filepath": "path/to/audio2.wav", "duration": 3.45, "text": "third transcript in this example"}

.. note::
    Only ``audio_filepath`` is field mandatory. If ``duration`` is not specified, the whole audio file will be used. A missing ``text`` field
    will produce an empty string as a text.

.. warning::
    Handling of ``duration`` and ``offset`` fields is not yet implemented. The current implementation always reads the whole audio file.

This reader produces between 1 and 3 outputs:

- Decoded audio data: float, ``shape=(audio_length,)``
- (optional, if ``read_sample_rate=True``) Audio sample rate: float, ``shape=(1,)``
- (optional, if ``read_text=True``) Transcript text as a null terminated string: uint8, ``shape=(text_len + 1,)``
- (optional, if ``read_idxs=True``) Index of the manifest entry: int64, ``shape=(1,)``

)code")
  .AddArg("manifest_filepaths",
    "List of paths to NeMo's compatible manifest files.",
    DALI_STRING_VEC)
  .AddOptionalArg("read_sample_rate",
    "Whether to output the sample rate for each sample as a separate output",
    true)
  .AddOptionalArg("read_text",
    "Whether to output the transcript text for each sample as a separate output",
    true)
  .AddOptionalArg("read_idxs",
    R"code(Whether to output the indices of samples as they occur in the manifest file
 as a separate output)code",
    false)
  .AddOptionalArg("shuffle_after_epoch",
    "If true, reader shuffles whole dataset after each epoch",
    false)
  .AddOptionalArg("sample_rate",
    "If specified, the target sample rate, in Hz, to which the audio is resampled.",
    -1.0f)
  .AddOptionalArg("quality",
    R"code(Resampling quality, 0 is lowest, 100 is highest.

  0 corresponds to 3 lobes of the sinc filter; 50 gives 16 lobes and 100 gives 64 lobes.)code",
     50.0f)
  .AddOptionalArg("downmix",
    "If True, downmix all input channels to mono. "
    "If downmixing is turned on, decoder will produce always 1-D output",
    true)
  .AddOptionalArg("dtype",
    R"code(Output data type.

Supported types: ``INT16``, ``INT32``, and ``FLOAT``.)code",
    DALI_FLOAT)
  .AddOptionalArg("min_duration",
    R"code(If a value greater than 0 is provided, it specifies the minimum allowed duration,
 in seconds, of the audio samples.

Samples with a duration shorter than this value will be ignored.)code",
    0.0f)
  .AddOptionalArg("max_duration",
    R"code(If a value greater than 0 is provided, it specifies the maximum allowed duration,
in seconds, of the audio samples.

Samples with a duration longer than this value will be ignored.)code",
    0.0f)
  .AddOptionalArg<bool>("normalize_text", "Normalize text.", nullptr)
  .DeprecateArg("normalize_text")  // deprecated since 0.28dev
  .AdditionalOutputsFn(NemoAsrReaderOutputFn)
  .AddParent("LoaderBase");


// Deprecated alias
DALI_REGISTER_OPERATOR(NemoAsrReader, NemoAsrReader, CPU);

DALI_SCHEMA(NemoAsrReader)
    .NumInput(0)
    .NumOutput(1)
    .DocStr("Legacy alias for :meth:`readers.nemo_asr`.")
    .AdditionalOutputsFn(NemoAsrReaderOutputFn)
    .AddParent("readers__NemoAsr")
    .MakeDocPartiallyHidden()
    .Deprecate(
        "readers__NemoAsr",
        R"code(In DALI 1.0 all readers were moved into a dedicated :mod:`~nvidia.dali.fn.readers`
submodule and renamed to follow a common pattern. This is a placeholder operator with identical
functionality to allow for backward compatibility.)code");  // Deprecated in 1.0;

NemoAsrReader::NemoAsrReader(const OpSpec& spec)
    : DataReader<CPUBackend, AsrSample>(spec),
      read_sr_(spec.GetArgument<bool>("read_sample_rate")),
      read_text_(spec.GetArgument<bool>("read_text")),
      read_idxs_(spec.GetArgument<bool>("read_idxs")),
      dtype_(spec.GetArgument<DALIDataType>("dtype")),
      num_threads_(std::max(1, spec.GetArgument<int>("num_threads"))),
      thread_pool_(num_threads_, spec.GetArgument<int>("device_id"), false) {
  loader_ = InitLoader<NemoAsrLoader>(spec);

  prefetched_decoded_audio_.resize(prefetch_queue_depth_);
  for (auto& batch : prefetched_decoded_audio_) {
    batch = std::make_unique<TensorVector<CPUBackend>>();
    batch->set_pinned(false);
  }
}

NemoAsrReader::~NemoAsrReader() {
  // Need to stop the prefetch thread before destroying the thread pool
  DataReader<CPUBackend, AsrSample>::StopPrefetchThread();
}

void NemoAsrReader::Prefetch() {
  DomainTimeRange tr("[DALI][NemoAsrReader] Prefetch #" + to_string(curr_batch_producer_),
                      DomainTimeRange::kRed);
  DataReader<CPUBackend, AsrSample>::Prefetch();
  auto &curr_batch = prefetched_batch_queue_[curr_batch_producer_];
  auto &audio_batch = *prefetched_decoded_audio_[curr_batch_producer_];
  int nsamples = static_cast<int>(curr_batch.size());

  assert(nsamples > 0);
  auto ndim = curr_batch[0]->shape().sample_dim();
  TensorListShape<> shape(nsamples, ndim);
  for (int i = 0; i < nsamples; i++) {
    shape.set_tensor_shape(i, curr_batch[i]->shape());
  }
  audio_batch.Resize(shape, TypeTable::GetTypeInfo(dtype_));

  // Waiting until all the audio samples are ready to be consumed
  decoded_map_.clear();
  for (int i = 0; i < nsamples; i++) {
    auto &sample = *curr_batch[i];
    auto &audio = audio_batch[i];

    if (decoded_map_.find(&sample) != decoded_map_.end())
      continue;
    decoded_map_[&sample] = i;

    const auto &audio_meta = sample.audio_meta();
    int64_t priority = audio_meta.length * audio_meta.channels;
    thread_pool_.AddWork(
      [&audio, &sample](int tid) {
        sample.decode_audio(audio, tid);
      }, priority);
  }
  thread_pool_.RunAll();

  if (decoded_map_.size() < static_cast<size_t>(nsamples)) {  // there are repeated samples
    for (int i = 0; i < nsamples; i++) {
      auto it = decoded_map_.find(curr_batch[i].get());
      if (it != decoded_map_.end() && it->second != i) {
        audio_batch[i].Copy(audio_batch[it->second], 0);
      }
    }
  }
}

void NemoAsrReader::RunImpl(SampleWorkspace &ws) {
  const auto &sample = GetSample(ws.data_idx());
  const auto &sample_audio = GetDecodedAudioSample(ws.data_idx());

  auto &audio = ws.Output<CPUBackend>(0);
  audio.Copy(sample_audio, 0);

  DALIMeta meta;
  meta.SetSourceInfo(sample.audio_filepath());
  meta.SetSkipSample(false);
  audio.SetMeta(meta);

  int next_out_idx = 1;
  if (read_sr_) {
    auto &sample_rate = ws.Output<CPUBackend>(next_out_idx++);
    sample_rate.Resize({});
    sample_rate.set_type(TypeTable::GetTypeInfo(DALI_FLOAT));
    sample_rate.mutable_data<float>()[0] = sample.audio_meta().sample_rate;
    sample_rate.SetMeta(meta);
  }

  if (read_text_) {
    auto &text_out = ws.Output<CPUBackend>(next_out_idx++);
    text_out.set_type(TypeTable::GetTypeInfo(DALI_UINT8));
    const auto &text = sample.text();
    int64_t text_sz = text.length();
    text_out.Resize({text_sz});
    std::memcpy(text_out.mutable_data<uint8_t>(), text.c_str(), text_sz);
    text_out.SetMeta(meta);
  }

  if (read_idxs_) {
    auto &idxs_out = ws.Output<CPUBackend>(next_out_idx++);
    idxs_out.set_type(TypeTable::GetTypeInfo(DALI_INT64));
    idxs_out.Resize({1});
    *idxs_out.mutable_data<int64_t>() = sample.index();
    idxs_out.SetMeta(meta);
  }
}

Tensor<CPUBackend>& NemoAsrReader::GetDecodedAudioSample(int sample_idx) {
  auto &curr_batch = *prefetched_decoded_audio_[curr_batch_consumer_];
  return curr_batch[sample_idx];
}

}  // namespace dali
