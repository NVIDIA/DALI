from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import scipy.io.wavfile
import numpy as np
import math
import json

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

def create_wav_files():
  for i in range(len(names)):
    wave = generate_waveforms(lengths[i], freqs[i])
    wave = (wave * 32767).round().astype(np.int16)
    scipy.io.wavfile.write(names[i], rates[i], wave)

create_wav_files()

nemo_asr_manifest = "/tmp/nemo_asr_manifest.json"

def create_manifest_file():
  entry0 = {}
  entry0["audio_filepath"] = "/tmp/dali_test_1C.wav"
  entry0["duration"] = lengths[0] * (1.0 / rates[0])
  entry0["text"] = "dali test 1C"
  entry1 = {}
  entry1["audio_filepath"] = "/tmp/dali_test_2C.wav"
  entry0["duration"] = lengths[1] * (1.0 / rates[1])
  entry0["text"] = "dali test 2C"
  entry2 = {}
  entry2["audio_filepath"] = "/tmp/dali_test_4C.wav"
  entry0["duration"] = lengths[2] * (1.0 / rates[2])
  entry0["text"] = "dali test 4C"

  data = [entry0, entry1, entry2]
  with open(nemo_asr_manifest, 'w') as f:
    json.dump(data, f)

create_manifest_file()

rate1 = 16000
rate2 = 12999

class NemoAsrReaderPipeline(Pipeline):
  def __init__(self):
    super(NemoAsrReaderPipeline, self).__init__(batch_size=8, num_threads=3, device_id=0,
                                                exec_async=True, exec_pipelined=True)
    fixed_seed = 12345
    self.reader1 = ops.NemoAsrReader(manifest_filepath = nemo_asr_manifest, dtype = types.INT16, downmix = False, seed=fixed_seed)
    self.reader2 = ops.NemoAsrReader(manifest_filepath = nemo_asr_manifest, dtype = types.INT16, downmix = True, seed=fixed_seed)
    #self.reader3 = ops.NemoAsrReader(manifest_filepath = nemo_asr_manifest, dtype = types.INT16, downmix = True,
    #                                 sample_rate=rate1, read_sample_rate=True, seed=fixed_seed)
    #self.reader4 = ops.NemoAsrReader(manifest_filepath = nemo_asr_manifest, dtype = types.INT16, downmix = True,
    #                                 sample_rate=rate2, read_sample_rate=True, seed=fixed_seed)

  def define_graph(self):
    audio_plain = self.reader1()
#    audio_downmixed = self.reader2()
#    audio_resampled1, sr1 = self.reader3()
#    audio_resampled2, sr2 = self.reader4()
    return audio_plain

def test_decoded_vs_generated():
  pipeline = NemoAsrReaderPipeline()
  pipeline.build()
  idx = 0
  for iter in range(1):
    out = pipeline.run()
    for i in range(len(out[0])):
      plain = out[0].at(i)
test_decoded_vs_generated()