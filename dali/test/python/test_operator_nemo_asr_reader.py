from nvidia.dali.pipeline import Pipeline
from nvidia.dali import fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import scipy.io.wavfile
import numpy as np
import math
import json
import librosa
import tempfile
import os
from test_audio_decoder_utils import generate_waveforms, rosa_resample

tmp_dir = tempfile.TemporaryDirectory()

names = [
  os.path.join(tmp_dir.name, "dali_test_1C.wav"),
  os.path.join(tmp_dir.name, "dali_test_2C.wav"),
  os.path.join(tmp_dir.name, "dali_test_4C.wav")
]

freqs = [
  np.array([0.02]),
  np.array([0.01, 0.012]),
  np.array([0.01, 0.012, 0.013, 0.014])
]
rates = [ 22050, 22050, 12347 ]
lengths = [ 10000, 54321, 12345 ]

def create_ref():
  ref = []
  for i in range(len(names)):
    wave = generate_waveforms(lengths[i], freqs[i])
    wave = (wave * 32767).round().astype(np.int16)
    ref.append(wave)
  return ref

ref_i = create_ref()

ref_text = [
  np.array([100, 97, 108, 105, 32, 116, 101, 115, 116, 32, 49, 67, 0], dtype=np.uint8),
  np.array([100, 97, 108, 105, 32, 116, 101, 115, 116, 32, 50, 67, 0], dtype=np.uint8),
  np.array([100, 97, 108, 105, 32, 116, 101, 115, 116, 32, 52, 67, 0], dtype=np.uint8)
]

def create_wav_files():
  for i in range(len(names)):
    scipy.io.wavfile.write(names[i], rates[i], ref_i[i])

create_wav_files()

nemo_asr_manifest = os.path.join(tmp_dir.name, "nemo_asr_manifest.json")

def create_manifest_file():
  entry0 = {}
  entry0["audio_filepath"] = names[0]
  entry0["duration"] = lengths[0] * (1.0 / rates[0])
  entry0["text"] = "dali test 1C"
  entry1 = {}
  entry1["audio_filepath"] = names[1]
  entry1["duration"] = lengths[1] * (1.0 / rates[1])
  entry1["text"] = "dali test 2C"
  entry2 = {}
  entry2["audio_filepath"] = names[2]
  entry2["duration"] = lengths[2] * (1.0 / rates[2])
  entry2["text"] = "dali test 4C"

  data = [entry0, entry1, entry2]
  with open(nemo_asr_manifest, 'w') as f:
    for entry in data:
      json.dump(entry, f)
      f.write('\n')

create_manifest_file()

rate1 = 16000
rate2 = 44100

class NemoAsrReaderPipeline(Pipeline):
  def __init__(self, batch_size=8):
    super(NemoAsrReaderPipeline, self).__init__(batch_size=batch_size, num_threads=1, device_id=0,
                                                exec_async=True, exec_pipelined=True)

  def define_graph(self):
    fixed_seed = 12345
    audio_plain_i = fn.nemo_asr_reader(manifest_filepaths = [nemo_asr_manifest], dtype = types.INT16, downmix = False,
                                       read_sample_rate = False, read_text = False, seed=fixed_seed)
    audio_plain_f = fn.nemo_asr_reader(manifest_filepaths = [nemo_asr_manifest], dtype = types.FLOAT, downmix = False,
                                       read_sample_rate = False, read_text = False, seed=fixed_seed)
    audio_downmix_i = fn.nemo_asr_reader(manifest_filepaths = [nemo_asr_manifest], dtype = types.INT16, downmix = True,
                                         read_sample_rate=False, read_text=False, seed=fixed_seed)
    audio_downmix_f = fn.nemo_asr_reader(manifest_filepaths = [nemo_asr_manifest], dtype = types.FLOAT, downmix = True,
                                         read_sample_rate=False, read_text=False, seed=fixed_seed)
    audio_resampled1_i, sr1_i = fn.nemo_asr_reader(manifest_filepaths = [nemo_asr_manifest], dtype = types.INT16, downmix = True,
                                                   sample_rate=rate1, read_sample_rate=True, read_text=False, seed=fixed_seed)
    audio_resampled1_f, sr1_f = fn.nemo_asr_reader(manifest_filepaths = [nemo_asr_manifest], dtype = types.FLOAT, downmix = True,
                                                   sample_rate=rate1, read_sample_rate=True, read_text=False, seed=fixed_seed)
    audio_resampled2_i, sr1_i = fn.nemo_asr_reader(manifest_filepaths = [nemo_asr_manifest], dtype = types.INT16, downmix = True,
                                                   sample_rate=rate2, read_sample_rate=True, read_text=False, seed=fixed_seed)
    audio_resampled2_f, sr1_f = fn.nemo_asr_reader(manifest_filepaths = [nemo_asr_manifest], dtype = types.FLOAT, downmix = True,
                                                   sample_rate=rate2, read_sample_rate=True, read_text=False, seed=fixed_seed)
    _, _, text = fn.nemo_asr_reader(manifest_filepaths = [nemo_asr_manifest], dtype = types.INT16, downmix = True,
                                    read_sample_rate=True, read_text=True, seed=fixed_seed)
    return audio_plain_i, audio_plain_f, audio_downmix_i, audio_downmix_f, \
           audio_resampled1_i, audio_resampled1_f, audio_resampled2_i, audio_resampled2_f, \
           text

def test_decoded_vs_generated(batch_size=3):
  pipeline = NemoAsrReaderPipeline(batch_size=batch_size)
  pipeline.build()

  for iter in range(1):
    out = pipeline.run()
    for idx in range(batch_size):
      audio_plain_i = out[0].at(idx)
      audio_plain_f = out[1].at(idx)
      audio_downmix_i = out[2].at(idx)
      audio_downmix_f = out[3].at(idx)
      audio_resampled1_i = out[4].at(idx)
      audio_resampled1_f = out[5].at(idx)
      audio_resampled2_i = out[6].at(idx)
      audio_resampled2_f = out[7].at(idx)
      text = out[8].at(idx)

      ref_plain_i = ref_i[idx]
      np.testing.assert_allclose(audio_plain_i, ref_plain_i, rtol = 1e-7)

      ref_plain_f = ref_i[idx].astype(np.float32) / 32767
      np.testing.assert_allclose(audio_plain_f, ref_plain_f, rtol = 1e-4)

      ref_downmix_i_float = ref_i[idx].astype(np.float32).mean(axis = 1, keepdims = 1)

      ref_downmix_i = ref_downmix_i_float.astype(np.int16).flatten()
      np.testing.assert_allclose(audio_downmix_i, ref_downmix_i, atol = 1)

      ref_downmix_f = (ref_downmix_i_float / 32767).flatten()
      np.testing.assert_allclose(audio_downmix_f, ref_downmix_f, rtol = 1e-4)

      ref_resampled1_float = generate_waveforms(lengths[idx] * rate1 / rates[idx], freqs[idx] * (rates[idx] / rate1))
      ref_resampled1_downmix = ref_resampled1_float.astype(np.float32).mean(axis = 1, keepdims = 1)
      ref_resampled1_i = (ref_resampled1_downmix * 32767).astype(np.int16).flatten()
      # resampling - allow for 1e-3 dynamic range error
      np.testing.assert_allclose(audio_resampled1_i, ref_resampled1_i, atol=round(32767 * 1e-3))

      ref_resampled1_f = ref_resampled1_downmix.flatten()
      # resampling - allow for 1e-3 dynamic range error
      np.testing.assert_allclose(audio_resampled1_f, ref_resampled1_f, atol=1e-3)

      ref_resampled2_float = generate_waveforms(lengths[idx] * rate2 / rates[idx], freqs[idx] * (rates[idx] / rate2))
      ref_resampled2_downmix = ref_resampled2_float.astype(np.float32).mean(axis = 1, keepdims = 1)
      ref_resampled2_i = (ref_resampled2_downmix * 32767).astype(np.int16).flatten()
      # resampling - allow for 1e-3 dynamic range error
      np.testing.assert_allclose(audio_resampled2_i, ref_resampled2_i, atol=round(32767 * 1e-3))

      ref_resampled2_f = ref_resampled2_downmix.flatten()
      # resampling - allow for 1e-3 dynamic range error
      np.testing.assert_allclose(audio_resampled2_f, ref_resampled2_f, atol=1e-3)

      np.testing.assert_equal(text, ref_text[idx])
