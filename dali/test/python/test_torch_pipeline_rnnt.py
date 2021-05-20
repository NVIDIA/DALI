# Copyright (c) 2020, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import nvidia.dali
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import test_utils
import numpy as np
import librosa
import librosa
import torch
import math
import os

audio_files = test_utils.get_files('db/audio/wav', 'wav')
audio_files = [file for file in audio_files if '237-134500' in file]  # Filtering librispeech samples
npy_files = [os.path.splitext(fpath)[0] + '.npy' for fpath in audio_files]

# From DeepLearningExamples
def _convert_samples_to_float32(samples):
    """Convert sample type to float32.
    Audio sample type is usually integer or float-point.
    Integers will be scaled to [-1, 1] in float32.
    """
    float32_samples = samples.astype('float32')
    if samples.dtype in np.sctypes['int']:
        bits = np.iinfo(samples.dtype).bits
        float32_samples *= (1. / 2 ** (bits - 1))
    elif samples.dtype in np.sctypes['float']:
        pass
    else:
        raise TypeError("Unsupported sample type: %s." % samples.dtype)
    return float32_samples

class FilterbankFeatures():
    def __init__(self, sample_rate=8000, window_size=0.02, window_stride=0.01,
                 window="hann", normalize="per_feature", n_fft=None,
                 preemph=0.97,
                 nfilt=64, lowfreq=0, highfreq=None, log=True, dither=.00001,
                 pad_to=8,
                 max_duration=16.7,
                 frame_splicing=1):

        torch_windows = {
            'hann': torch.hann_window,
            'hamming': torch.hamming_window,
            'blackman': torch.blackman_window,
            'bartlett': torch.bartlett_window,
            'none': None,
        }

        self.win_length = int(sample_rate * window_size)  # frame size
        self.hop_length = int(sample_rate * window_stride)
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))

        self.normalize = normalize
        self.log = log
        self.dither = dither
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        self.pad_to = pad_to
        highfreq = highfreq or sample_rate / 2
        window_fn = torch_windows.get(window, None)
        window_tensor = window_fn(self.win_length,
                                  periodic=False) if window_fn else None
        filterbanks = torch.tensor(
            librosa.filters.mel(sample_rate, self.n_fft, n_mels=nfilt, fmin=lowfreq,
                                fmax=highfreq), dtype=torch.float).unsqueeze(0)
        self.fb = filterbanks
        self.window = window_tensor
        # self.register_buffer("fb", filterbanks)
        # self.register_buffer("window", window_tensor)
        # Calculate maximum sequence length (# frames)
        max_length = 1 + math.ceil(
            (max_duration * sample_rate - self.win_length) / self.hop_length
        )
        max_pad = 16 - (max_length % 16)
        self.max_length = max_length + max_pad

    @staticmethod
    def splice_frames(x, frame_splicing):
        """ Stacks frames together across feature dim

        input is batch_size, feature_dim, num_frames
        output is batch_size, feature_dim*frame_splicing, num_frames

        """
        seq = [x]
        for n in range(1, frame_splicing):
            tmp = torch.zeros_like(x)
            tmp[:, :, :-n] = x[:, :, n:]
            seq.append(tmp)
        return torch.cat(seq, dim=1)[:, :, ::frame_splicing]

    @staticmethod
    def normalize_batch(x, seq_len, normalize_type):
        constant = 1e-5
        if normalize_type == "per_feature":
            x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype,
                                 device=x.device)
            x_std = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype,
                                device=x.device)
            for i in range(x.shape[0]):
                x_mean[i, :] = x[i, :, :seq_len[i]].mean(dim=1)
                x_std[i, :] = x[i, :, :seq_len[i]].std(dim=1)
            # make sure x_std is not zero
            x_std += constant
            return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
        elif normalize_type == "all_features":
            x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
            x_std = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
            for i in range(x.shape[0]):
                x_mean[i] = x[i, :, :seq_len[i].item()].mean()
                x_std[i] = x[i, :, :seq_len[i].item()].std()
            # make sure x_std is not zero
            x_std += constant
            return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1)
        else:
            return x

    def get_seq_len(self, seq_len):
        x = torch.ceil(seq_len.to(dtype=torch.float) / self.hop_length).to(
            dtype=torch.int)
        if self.frame_splicing > 1:
            x = torch.ceil(x.float() / self.frame_splicing).to(dtype=torch.int)
        return x

    def forward(self, inp, seq_len):
        x = inp
        dtype = x.dtype

        seq_len = self.get_seq_len(seq_len)

        # do preemphasis
        if self.preemph is not None:
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]),
                          dim=1)

        # do stft
        x = torch.stft(x, n_fft=self.n_fft, hop_length=self.hop_length,
                       win_length=self.win_length,
                       center=True, window=self.window.to(dtype=torch.float))

        # get power spectrum
        x = x.pow(2).sum(-1)

        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)

        # log features if required
        if self.log:
            x = torch.log(x + 1e-20)

        # frame splicing if required
        if self.frame_splicing > 1:
            x = self.splice_frames(x, self.frame_splicing)

        # normalize if required
        if self.normalize:
            x = self.normalize_batch(x, seq_len, normalize_type=self.normalize)

        return x.to(dtype)


class RnntTrainPipeline(nvidia.dali.Pipeline):
    def __init__(self,
                 device_id,
                 n_devices,
                 files,
                 batch_size,
                 sample_rate=16000,
                 window_size=.02,
                 window_stride=.01,
                 nfeatures=64,
                 nfft=512,
                 frame_splicing_factor=3,
                 silence_threshold=-80,
                 dither=.00001,
                 preemph_coeff=.97,
                 lowfreq=0.0,
                 highfreq=0.0,
                 num_threads=1):
        super().__init__(batch_size, num_threads, device_id, seed=42)

        self.dither = dither
        self.frame_splicing_factor = frame_splicing_factor

        self.read = ops.readers.File(files=files, device="cpu", random_shuffle=False,
                                     shard_id=device_id, num_shards=n_devices)

        self.decode = ops.decoders.Audio(device="cpu", dtype=types.FLOAT, downmix=True)

        self.normal_distribution = ops.random.Normal(device="cpu")

        self.preemph = ops.PreemphasisFilter(preemph_coeff=preemph_coeff)

        self.spectrogram = ops.Spectrogram(device="cpu", nfft=nfft,
                                           window_length=window_size * sample_rate,
                                           window_step=window_stride * sample_rate)

        self.mel_fbank = ops.MelFilterBank(device="cpu", sample_rate=sample_rate, nfilter=nfeatures,
                                           normalize=True, freq_low=lowfreq, freq_high=highfreq)

        self.log_features = ops.ToDecibels(device="cpu", multiplier=np.log(10), reference=1.0,
                                           cutoff_db=-80)

        self.get_shape = ops.Shapes(device="cpu")

        self.normalize = ops.Normalize(axes=[0], device="cpu")

        self.splicing_transpose = ops.Transpose(device="cpu", perm=[1, 0])
        self.splicing_reshape = ops.Reshape(device="cpu", rel_shape=[-1, frame_splicing_factor])
        self.splicing_pad = ops.Pad(axes=[0], fill_value=0, align=frame_splicing_factor, shape=[1],
                                    device="cpu")

        self.get_nonsilent_region = ops.NonsilentRegion(device="cpu", cutoff_db=silence_threshold)
        self.trim_silence = ops.Slice(device="cpu", axes=[0])
        self.to_float = ops.Cast(dtype=types.FLOAT)

    @staticmethod
    def _div_ceil(dividend, divisor):
        return (dividend + divisor - 1) // divisor

    def _splice_frames(self, input):
        """
        Frame splicing is implemented by transposing the input, padding it and reshaping.
        Theoretically, to achieve the result now there should be one more transpose at the end,
        but it can be skipped as an optimization
        """
        out = self.splicing_transpose(input)

        # Because of the padding, we need to determine length of audio sample before it occurs
        audio_len = self._div_ceil(self.get_shape(out), self.frame_splicing_factor)

        out = self.splicing_pad(out)
        out = self.splicing_reshape(out)
        # Skipping transposing back
        return out, audio_len

    def remove_silence(self, input):
        begin, len = self.get_nonsilent_region(input)
        out = self.trim_silence(input, begin, len)
        return out

    def define_graph(self):
        input, label = self.read()
        decoded, sr = self.decode(input)

        audio = self.remove_silence(decoded)

        # DALI's preemph works a little bit different than the one in native code.
        # The difference occurs in first value in buffer.
        audio = self.preemph(audio)

        audio = self.spectrogram(audio)
        audio = self.mel_fbank(audio)
        audio = self.log_features(audio)
        audio, audio_sh = self._splice_frames(audio)

        # This normalization goes across ax=0, since
        # the frame splicing doesn't transpose the tensor back
        audio = self.normalize(audio)

        return audio, audio_sh


def test_rnnt_data_pipeline():
    """
    Test compares pre-calculated output of native data pipeline with an output
    from DALI data pipeline. There are few modification of native data pipeline
    comparing to the reference: random operations (i.e. dither and presampling
    aka "speed perturbation") are turned off
    """
    ref_pipeline = FilterbankFeatures(sample_rate=16000, n_fft=512, highfreq=.0, dither=.00001,
                                      frame_splicing=3)
    recordings = []
    for fpath in npy_files:
        arr = np.load(fpath)
        arr = _convert_samples_to_float32(arr)
        recordings.append(arr)
    batch_size = len(recordings)
    pipe = RnntTrainPipeline(device_id=0, n_devices=1, files=audio_files, batch_size=batch_size)
    pipe.build()
    dali_out = pipe.run()
    reference_data = []
    for i in range(batch_size):
        reference_data.append(
            ref_pipeline.forward(torch.tensor([recordings[i]]), torch.tensor([recordings[i].shape[0]]))
        )
    for sample_idx in range(batch_size):
        output_data = dali_out[0].at(sample_idx)
        output_data = np.transpose(output_data, (1, 0))
        audio_len = dali_out[1].at(sample_idx)[0]
        assert audio_len == reference_data[sample_idx].shape[2]
        nfeatures = reference_data[sample_idx].shape[1]
        assert nfeatures == output_data.shape[0]
        size = nfeatures * audio_len
        ref = reference_data[sample_idx][:, :, :]
        out = output_data[:, :]
        assert np.sum(np.isclose(ref, out, atol=.1, rtol=0)) / size > .9, \
            f"{np.sum(np.isclose(ref, out, atol=.1, rtol=0)) / size}"
