# ISC License
# Copyright (c) 2013--2023, librosa development team.

# Permission to use, copy, modify, and/or distribute this software for any purpose
# with or without fee is hereby granted, provided that the above copyright notice
# and this permission notice appear in all copies.

# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
# REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY
# AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
# INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
# LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE
# OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
# PERFORMANCE OF THIS SOFTWARE.

# DALI team note:
#   The code above has been copied and adapted from Librosa, and it is meant
#   to be used as a reference for testing purposes.

import numpy as np
import scipy


def power_to_db(magnitude, ref=1.0, amin=1e-10, top_db=80.0):
    if amin <= 0:
        raise RuntimeError("amin must be strictly positive")

    if callable(ref):
        # User supplied a function to calculate reference power
        ref_value = ref(magnitude)
    else:
        ref_value = np.abs(ref)

    log_spec = 10.0 * np.log10(np.maximum(amin, magnitude))
    log_spec -= 10.0 * np.log10(np.maximum(amin, ref_value))

    if top_db is not None:
        if top_db < 0:
            raise RuntimeError("top_db must be non-negative")
        log_spec = np.maximum(log_spec, log_spec.max() - top_db)

    return log_spec


def extract_frames(x, frame_length, hop_length, axis=-1):
    assert (
        x.shape[axis] >= frame_length
    ), f"Input length should be <= frame_length: Got {x.shape[axis]} < {frame_length}"
    assert hop_length >= 1, f"Invalid hop_length: {hop_length}"
    n_frames = 1 + (x.shape[axis] - frame_length) // hop_length
    x_frames = np.zeros((frame_length, n_frames), dtype=np.float32)
    for j in range(n_frames):
        x_frames[:, j] = x[j * hop_length : j * hop_length + frame_length]
    return x_frames


def nonsilent_region(y, top_db, ref, frame_length, hop_length):
    # Convert to mono, if needed
    if y.ndim > 1:
        y = np.mean(y, axis=0)

    # Centered windows
    y = np.pad(y, int(frame_length // 2), mode="reflect")

    # Extract overlapping frames. Shape: (frame_length, num_frames)
    frames = extract_frames(y, frame_length=frame_length, hop_length=hop_length)

    # Calculate power
    mse = np.mean(np.abs(frames) ** 2, axis=0, keepdims=True)

    non_silent_frames = power_to_db(mse.squeeze(), ref=ref, top_db=None) > -top_db

    start, end = 0, 0
    nonzero = np.flatnonzero(non_silent_frames)
    if nonzero.size > 0:
        # Compute the start and end positions
        # End position goes one frame past the last non-zero
        start = int(nonzero[0] * hop_length)
        end = min(y.shape[-1], int((nonzero[-1] + 1) * hop_length))

    # librosa's trim function calculates power with reference to center of window,
    # while DALI uses beginning of window. Hence the subtraction below
    length = end - start
    if length != 0:
        length += frame_length - 1
    start = start - frame_length // 2

    return (start, length)


def pad_center(data, *, size, axis=-1, **kwargs):
    kwargs.setdefault("mode", "constant")

    n = data.shape[axis]

    lpad = int((size - n) // 2)

    lengths = [(0, 0)] * data.ndim
    lengths[axis] = (lpad, int(size - n - lpad))

    if lpad < 0:
        raise RuntimeError(
            ("Target size ({:d}) must be " "at least input size ({:d})").format(size, n)
        )

    return np.pad(data, lengths, **kwargs)


def expand_to(x, *, ndim, axes):
    # Force axes into a tuple

    try:
        axes = tuple(axes)
    except TypeError:
        axes = tuple([axes])

    if len(axes) != x.ndim:
        raise RuntimeError(
            "Shape mismatch between axes={} and input x.shape={}".format(axes, x.shape)
        )

    if ndim < x.ndim:
        raise RuntimeError(
            "Cannot expand x.shape={} to fewer dimensions ndim={}".format(x.shape, ndim)
        )

    shape = [1] * ndim
    for i, axi in enumerate(axes):
        shape[axi] = x.shape[i]

    return x.reshape(shape)


def dtype_r2c(d, *, default=np.complex64):
    mapping = {
        np.dtype(np.float32): np.complex64,
        np.dtype(np.float64): np.complex128,
        np.dtype(float): np.dtype(complex).type,
    }

    # If we're given a complex type already, return it
    dt = np.dtype(d)
    if dt.kind == "c":
        return dt

    # Otherwise, try to map the dtype.
    # If no match is found, return the default.
    return np.dtype(mapping.get(dt, default))


def get_window(window, Nx, *, fftbins=True):
    if callable(window):
        return window(Nx)

    elif isinstance(window, (str, tuple)) or np.isscalar(window):
        # TODO: if we add custom window functions in librosa, call them here

        return scipy.signal.get_window(window, Nx, fftbins=fftbins)

    elif isinstance(window, (np.ndarray, list)):
        if len(window) == Nx:
            return np.asarray(window)

        raise RuntimeError("Window size mismatch: " "{:d} != {:d}".format(len(window), Nx))
    else:
        raise RuntimeError("Invalid window specification: {}".format(window))


def stft(
    y,
    *,
    n_fft=2048,
    hop_length=None,
    win_length=None,
    window="hann",
    center=True,
    dtype=None,
    pad_mode="reflect",
):
    # By default, use the entire frame
    if win_length is None:
        win_length = n_fft

    # Set the default hop, if it's not already specified
    if hop_length is None:
        hop_length = int(win_length // 4)

    fft_window = get_window(window, win_length, fftbins=True)

    # Pad the window out to n_fft size
    fft_window = pad_center(fft_window, size=n_fft)

    # Reshape so that the window can be broadcast
    fft_window = expand_to(fft_window, ndim=1 + y.ndim, axes=-2)

    # Pad the time series so that frames are centered
    if center:
        if n_fft > y.shape[-1]:
            print(
                "n_fft={} is too small for input signal of length={}".format(n_fft, y.shape[-1]),
                stacklevel=2,
            )

        padding = [(0, 0) for _ in range(y.ndim)]
        padding[-1] = (int(n_fft // 2), int(n_fft // 2))
        y = np.pad(y, padding, mode=pad_mode)

    elif n_fft > y.shape[-1]:
        raise RuntimeError(
            "n_fft={} is too large for input signal of length={}".format(n_fft, y.shape[-1])
        )

    # Window the time series.

    y_frames = extract_frames(y, frame_length=n_fft, hop_length=hop_length, axis=-1)

    if dtype is None:
        dtype = dtype_r2c(y.dtype)

    # Pre-allocate the STFT matrix
    shape = list(y_frames.shape)
    shape[-2] = 1 + n_fft // 2
    stft_matrix = np.empty(shape, dtype=dtype, order="F")

    # Constrain STFT block sizes to 256 KB
    MAX_MEM_BLOCK = 2**8 * 2**10
    n_columns = MAX_MEM_BLOCK // (np.prod(stft_matrix.shape[:-1]) * stft_matrix.itemsize)
    n_columns = max(n_columns, 1)

    for bl_s in range(0, stft_matrix.shape[-1], n_columns):
        bl_t = min(bl_s + n_columns, stft_matrix.shape[-1])

        stft_matrix[..., bl_s:bl_t] = np.fft.rfft(fft_window * y_frames[..., bl_s:bl_t], axis=-2)
    return stft_matrix
