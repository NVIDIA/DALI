from __future__ import print_function
from __future__ import division
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import scipy.io.wavfile
import numpy as np
import math
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
rates = [ 16000, 22050, 12347 ]
lengths = [ 10000, 54321, 12345 ]

def create_test_files():
  for i in range(len(names)):
    wave = generate_waveforms(lengths[i], freqs[i])
    wave = (wave * 32767).round().astype(np.int16)
    scipy.io.wavfile.write(names[i], rates[i], wave)


create_test_files()

rate1 = 16000
rate2 = 12999

class DecoderPipeline(Pipeline):
  def __init__(self):
    super(DecoderPipeline, self).__init__(batch_size=8, num_threads=3, device_id=0,
                                          exec_async=True, exec_pipelined=True)
    self.file_source = ops.ExternalSource()
    self.plain_decoder = ops.AudioDecoder(dtype = types.INT16)
    self.resampling_decoder = ops.AudioDecoder(sample_rate=rate1, dtype = types.INT16)
    self.downmixing_decoder = ops.AudioDecoder(downmix=True, dtype = types.INT16)
    self.resampling_downmixing_decoder = ops.AudioDecoder(sample_rate=rate2, downmix=True,
                                                          quality=50, dtype = types.FLOAT)

  def define_graph(self):
    self.raw_file = self.file_source()
    dec_plain, rates_plain = self.plain_decoder(self.raw_file)
    dec_res, rates_res = self.resampling_decoder(self.raw_file)
    dec_mix, rates_mix = self.downmixing_decoder(self.raw_file)
    dec_res_mix, rates_res_mix = self.resampling_downmixing_decoder(self.raw_file)
    out = [dec_plain, dec_res, dec_mix, dec_res_mix,
           rates_plain, rates_res, rates_mix, rates_res_mix]
    return out

  def iter_setup(self):
    list = []
    for i in range(self.batch_size):
      idx = i % len(names)
      with open(names[idx], mode = "rb") as f:
        list.append(np.array(bytearray(f.read()), np.uint8))
    self.feed_input(self.raw_file, list)


def rosa_resample(input, in_rate, out_rate):
  if input.shape[1] == 1:
    return librosa.resample(input[:,0], in_rate, out_rate)[:,np.newaxis]

  channels = [librosa.resample(np.array(input[:,c]), in_rate, out_rate) for c in range(input.shape[1])]
  ret = np.zeros(shape = [channels[0].shape[0], len(channels)], dtype=channels[0].dtype)
  for c, a in enumerate(channels):
    ret[:,c] = a

  return ret



def test_decoded_vs_generated():
  pipeline = DecoderPipeline()
  pipeline.build();
  idx = 0
  for iter in range(1):
    out = pipeline.run()
    for i in range(len(out[0])):
      plain = out[0].at(i)
      res = out[1].at(i)
      mix = out[2].at(i)[:,np.newaxis]
      res_mix = out[3].at(i)[:,np.newaxis]

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

      assert(out[4].at(i)[0] == rates[idx])
      assert(out[5].at(i)[0] == rate1)
      assert(out[6].at(i)[0] == rates[idx])
      assert(out[7].at(i)[0] == rate2)

      # just reading - allow only for rounding
      assert np.allclose(plain, ref0, rtol = 0, atol=0.5)
      # resampling - allow for 1e-3 dynamic range error
      assert np.allclose(res, ref1, rtol = 0, atol=32767 * 1e-3)
      # downmixing - allow for 2 bits of error
      # - one for quantization of channels, one for quantization of result
      assert np.allclose(mix, ref2, rtol = 0, atol=2)
      # resampling with weird ratio - allow for 3e-3 dynamic range error
      assert np.allclose(res_mix, ref3, rtol = 0, atol=3e-3)

      rosa_in1 = plain.astype(np.float32)
      rosa1 = rosa_resample(rosa_in1, rates[idx], rate1)
      rosa_in3 = rosa_in1 / 32767;
      rosa3 = rosa_resample(rosa_in3.mean(axis = 1, keepdims = 1), rates[idx], rate2)

      assert np.allclose(res, rosa1, rtol = 0, atol=32767 * 1e-3)
      assert np.allclose(res_mix, rosa3, rtol = 0, atol=3e-3)

      idx = (idx + 1) % len(names)

def main():
  test_decoded_vs_generated()


if __name__ == '__main__':
  main()
