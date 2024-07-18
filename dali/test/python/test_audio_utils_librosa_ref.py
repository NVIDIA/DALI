
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
    assert x.shape[axis] >= frame_length, \
        f"Input should be of equal length or higher than frame_length: Got {x.shape[axis]} < {frame_length}"
    assert hop_length >= 1, \
        f"Invalid hop_length: {hop_length}"
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
