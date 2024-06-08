import numpy as np
import math


def generate_waveforms(length, frequencies):
    """
    generate sinewaves with given frequencies,
    add Hann envelope and store in channel-last layout
    """
    n = int(math.ceil(length))
    X = np.arange(n, dtype=np.float32)

    def window(x):
        x = 2 * x / length - 1
        np.clip(x, -1, 1, out=x)
        return 0.5 * (1 + np.cos(x * math.pi))

    wave = np.sin(X[:, np.newaxis] * (np.array(frequencies) * (2 * math.pi)))
    return wave * window(X)[:, np.newaxis]
