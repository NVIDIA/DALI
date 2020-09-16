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

def rosa_resample(input, in_rate, out_rate):
  if input.shape[1] == 1:
    return librosa.resample(input[:,0], in_rate, out_rate)[:,np.newaxis]

  channels = [librosa.resample(np.array(input[:,c]), in_rate, out_rate) for c in range(input.shape[1])]
  ret = np.zeros(shape = [channels[0].shape[0], len(channels)], dtype=channels[0].dtype)
  for c, a in enumerate(channels):
    ret[:,c] = a

  return ret
