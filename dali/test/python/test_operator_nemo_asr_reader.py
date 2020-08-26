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

def create_wav_files():
  for i in range(len(names)):
    wave = generate_waveforms(lengths[i], freqs[i])
    wave = (wave * 32767).round().astype(np.int16)
    scipy.io.wavfile.write(names[i], rates[i], wave)

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
rate2 = 8000

class NemoAsrReaderPipeline(Pipeline):
  def __init__(self, batch_size=8):
    super(NemoAsrReaderPipeline, self).__init__(batch_size=batch_size, num_threads=3, device_id=0,
                                                exec_async=True, exec_pipelined=True)
    fixed_seed = 12345
    self.reader1 = ops.NemoAsrReader(manifest_filepath = nemo_asr_manifest, dtype = types.INT16, downmix = False,
                                     read_sample_rate = False, read_text = False, seed=fixed_seed)
    self.reader2 = ops.NemoAsrReader(manifest_filepath = nemo_asr_manifest, dtype = types.INT16, downmix = True,
                                     read_sample_rate = False, read_text = False, seed=fixed_seed)
    self.reader3 = ops.NemoAsrReader(manifest_filepath = nemo_asr_manifest, dtype = types.INT16, downmix = True,
                                     sample_rate=rate1, read_sample_rate=True, read_text=False, seed=fixed_seed)
    self.reader4 = ops.NemoAsrReader(manifest_filepath = nemo_asr_manifest, dtype = types.INT16, downmix = True,
                                     sample_rate=rate2, read_sample_rate=True, read_text=False, seed=fixed_seed)
    self.reader5 = ops.NemoAsrReader(manifest_filepath = nemo_asr_manifest, dtype = types.INT16, downmix = True,
                                     read_sample_rate=False, read_text=True, seed=fixed_seed)
    self.reader6 = ops.NemoAsrReader(manifest_filepath = nemo_asr_manifest, dtype = types.INT16, downmix = True,
                                     read_sample_rate=True, read_text=True, seed=fixed_seed)

  def define_graph(self):
    audio_plain = self.reader1()
    audio_downmixed = self.reader2()
    audio_resampled1, sr1 = self.reader3()
    audio_resampled2, sr2 = self.reader4()
    audio5, text5 = self.reader5()
    audio6, sr6, text6 = self.reader6()
    return audio_plain, audio_downmixed, audio_resampled1, audio_resampled2, audio5, audio6, text5, text6

def test_decoded_vs_generated(batch_size=1):
  pipeline = NemoAsrReaderPipeline(batch_size=batch_size)
  pipeline.build()
  idx = 0
  for iter in range(1):
    out = pipeline.run()
    for i in range(batch_size):
      print(i)
      audio_plain = out[0].at(i)
      audio_downmixed = out[1].at(i)
      audio_resampled1 = out[2].at(i)
      audio_resampled2 = out[3].at(i)
      audio5 = out[4].at(i)
      audio6 = out[5].at(i)
      text5 = out[6].at(i)
      text6 = out[7].at(i)

      ref_len = [0,0,0,0]
      ref_len[0] = lengths[idx]
      ref_len[1] = lengths[idx] * rate1 / rates[idx]
      ref_len[2] = lengths[idx]
      ref_len[3] = lengths[idx] * rate2 / rates[idx]

      ref0 = generate_waveforms(ref_len[0], freqs[idx]) * 32767
      ref1 = generate_waveforms(ref_len[1], freqs[idx] * (rates[idx] / rate1)) * 32767
      ref2 = generate_waveforms(ref_len[2], freqs[idx]) * 32767
      ref2 = ref2.mean(axis = 1, keepdims = 1)
      ref3 = generate_waveforms(ref_len[3], freqs[idx] * (rates[idx] / rate2))
      ref3 = ref3.mean(axis = 1, keepdims = 1)

      # just reading - allow only for rounding
      import sys
      #import numpy
      np.set_printoptions(threshold=sys.maxsize)
      ref_plain = ref0.round().astype(np.int16)
      assert np.allclose(audio_plain, ref_plain, rtol = 1e-7, atol=0.0)
      
      # resampling
      ref_resampled1 = rosa_resample(ref0, rates[idx], rate1).round().astype(np.int16).flatten()
      diff = ref_resampled1 - audio_resampled1
      print(audio_resampled1)
      print(diff.shape)
      print(diff.max())
      print(diff.mean())
      assert np.allclose(audio_resampled1, ref_resampled1, rtol = 1e-7, atol=0)

      ref_resampled2 = rosa_resample(ref0, rates[idx], rate2).round().astype(np.int16).flatten()
      diff = audio_resampled1
      print(diff)
      print(audio_resampled2.shape)
      print(ref_resampled2.shape)
      assert np.allclose(audio_resampled2, ref_resampled2, rtol = 0.1, atol=0)
      
      
      # downmixing - allow for 2 bits of error
      # - one for quantization of channels, one for quantization of result
      #assert np.allclose(mix, ref2, rtol = 0, atol=2)
      # resampling with weird ratio - allow for 3e-3 dynamic range error
      #assert np.allclose(res_mix, ref3, rtol = 0, atol=3e-3)


test_decoded_vs_generated()