from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import scipy.io.wavfile
import numpy as np
import math
import json
import librosa

# generate sinewaves with given frequencies,
# add Hann envelope and store in channel-last layout
def generate_waveforms(length, frequencies):
  n = int(math.ceil(length))
  X = np.arange(n, dtype=np.float32)
  def window(x):
    x = 2 * x / length - 1
    np.clip(x, -1, 1, out=x)
    return 0.5 * (1 + np.cos(x * math.pi))

  return np.sin(X[:,np.newaxis] * (np.array(frequencies) * (2 * math.pi))) * window(X)[:,np.newaxis]

def rosa_resample(input, in_rate, out_rate):
  if input.shape[1] == 1:
    return librosa.resample(input[:,0], in_rate, out_rate)[:,np.newaxis]

  channels = [librosa.resample(np.array(input[:,c]), in_rate, out_rate) for c in range(input.shape[1])]
  ret = np.zeros(shape = [channels[0].shape[0], len(channels)], dtype=channels[0].dtype)
  for c, a in enumerate(channels):
    ret[:,c] = a

  return ret

names = [
  "/tmp/dali_test_1C.wav",
  "/tmp/dali_test_2C.wav",
  "/tmp/dali_test_4C.wav"
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

ref = create_ref()

ref_text = [
  np.array([100, 97, 108, 105, 32, 116, 101, 115, 116, 32, 49, 67, 0], dtype=np.uint8),
  np.array([100, 97, 108, 105, 32, 116, 101, 115, 116, 32, 50, 67, 0], dtype=np.uint8),
  np.array([100, 97, 108, 105, 32, 116, 101, 115, 116, 32, 52, 67, 0], dtype=np.uint8)
]

def create_wav_files():
  for i in range(len(names)):
    scipy.io.wavfile.write(names[i], rates[i], ref[i])

create_wav_files()

nemo_asr_manifest = "/tmp/nemo_asr_manifest.json"

def create_manifest_file():
  entry0 = {}
  entry0["audio_filepath"] = names[0]
  entry0["duration"] = 100000000000000000000 #lengths[0] * (1.0 / rates[0])
  entry0["text"] = "dali test 1C"
  entry1 = {}
  entry1["audio_filepath"] = names[1]
  entry1["duration"] = 100000000000000000000 # lengths[1] * (1.0 / rates[1])
  entry1["text"] = "dali test 2C"
  entry2 = {}
  entry2["audio_filepath"] = names[2]
  entry2["duration"] = 100000000000000000000 # lengths[2] * (1.0 / rates[2])
  entry2["text"] = "dali test 4C"

  data = [entry0, entry1, entry2]
  with open(nemo_asr_manifest, 'w') as f:
    json.dump(data, f)

create_manifest_file()

rate1 = 16000
rate2 = 44100

class NemoAsrReaderPipeline(Pipeline):
  def __init__(self, batch_size=8):
    super(NemoAsrReaderPipeline, self).__init__(batch_size=batch_size, num_threads=3, device_id=0,
                                                exec_async=True, exec_pipelined=True)
    fixed_seed = 12345
    self.reader_plain = ops.NemoAsrReader(manifest_filepath = nemo_asr_manifest, dtype = types.INT16, downmix = False,
                                          read_sample_rate = False, read_text = False, seed=fixed_seed)
    self.reader_downmix = ops.NemoAsrReader(manifest_filepath = nemo_asr_manifest, dtype = types.INT16, downmix = True,
                                            read_sample_rate=False, read_text=False, seed=fixed_seed)
    self.reader_resample1 = ops.NemoAsrReader(manifest_filepath = nemo_asr_manifest, dtype = types.INT16, downmix = True,
                                              sample_rate=rate1, read_sample_rate=True, read_text=False, seed=fixed_seed)
    self.reader_resample2 = ops.NemoAsrReader(manifest_filepath = nemo_asr_manifest, dtype = types.INT16, downmix = True,
                                              sample_rate=rate2, read_sample_rate=True, read_text=False, seed=fixed_seed)
    self.reader_text = ops.NemoAsrReader(manifest_filepath = nemo_asr_manifest, dtype = types.INT16, downmix = True,
                                         read_sample_rate=True, read_text=True, seed=fixed_seed)

  def define_graph(self):
    audio_plain = self.reader_plain()
    audio_downmix = self.reader_downmix()
    audio_resampled1, sr1 = self.reader_resample1()
    audio_resampled2, sr1 = self.reader_resample2()
    _, _, text = self.reader_text()
#    return audio_plain, audio_downmixed, audio_resampled1, audio_resampled2, audio5, audio6, text5, text6
    return audio_plain, audio_downmix, audio_resampled1, audio_resampled2, text


def test_decoded_vs_generated(batch_size=3):
  pipeline = NemoAsrReaderPipeline(batch_size=batch_size)
  pipeline.build()

  for iter in range(1):
    out = pipeline.run()
    for idx in range(batch_size):
      audio_plain = out[0].at(idx)
      audio_downmix = out[1].at(idx)
      audio_resampled1 = out[2].at(idx)
      audio_resampled2 = out[3].at(idx)
      text = out[4].at(idx)

      ref_plain = ref[idx]
      np.testing.assert_allclose(audio_plain, ref_plain, rtol = 1e-7)

      ref_downmix_float = ref[idx].astype(np.float32).mean(axis = 1, keepdims = 1)
      ref_downmix = ref_downmix_float.astype(np.int16).flatten()
      np.testing.assert_allclose(audio_downmix, ref_downmix, atol = 1)

      ref_resampled1_float = generate_waveforms(lengths[idx] * rate1 / rates[idx], freqs[idx] * (rates[idx] / rate1)) * 32767
      ref_resampled1_downmix = ref_resampled1_float.astype(np.float32).mean(axis = 1, keepdims = 1)
      ref_resampled1 = ref_resampled1_downmix.astype(np.int16).flatten()
      # resampling - allow for 1e-3 dynamic range error
      np.testing.assert_allclose(audio_resampled1, ref_resampled1, atol=round(32767 * 1e-3))

      ref_resampled2_float = generate_waveforms(lengths[idx] * rate2 / rates[idx], freqs[idx] * (rates[idx] / rate2)) * 32767
      ref_resampled2_downmix = ref_resampled2_float.astype(np.float32).mean(axis = 1, keepdims = 1)
      ref_resampled2 = ref_resampled2_downmix.astype(np.int16).flatten()
      # resampling - allow for 1e-3 dynamic range error
      np.testing.assert_allclose(audio_resampled2, ref_resampled2, atol=round(32767 * 1e-3))

      np.testing.assert_equal(text, ref_text[idx])
