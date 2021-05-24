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
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.types as types
import test_utils
import numpy as np
import librosa
import torch
import math
import os

audio_files = test_utils.get_files('db/audio/wav', 'wav')
audio_files = [file for file in audio_files if '237-134500' in file]  # Filtering librispeech samples
npy_files = [os.path.splitext(fpath)[0] + '.npy' for fpath in audio_files]
npy_files_sr = 16000

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

torch_windows = {
    'hann': torch.hann_window,
    'hamming': torch.hamming_window,
    'blackman': torch.blackman_window,
    'bartlett': torch.bartlett_window,
    'none': None,
}

class FilterbankFeatures():
    def __init__(self, sample_rate=16000, window_size=0.02, window_stride=0.01, window="hann", normalize="per_feature",
                 n_fft=None, preemph=0.97, nfilt=64, lowfreq=0, highfreq=None, log=True, frame_splicing=1):
        self.win_length = int(sample_rate * window_size)
        self.hop_length = int(sample_rate * window_stride)
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))

        self.normalize = normalize
        self.log = log
        self.frame_splicing = frame_splicing
        self.nfilt = nfilt
        self.preemph = preemph
        window_fn = torch_windows.get(window, None)
        self.window = window_fn(self.win_length, periodic=False) if window_fn else None
        self.fb = torch.tensor(
            librosa.filters.mel(sample_rate, self.n_fft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq), dtype=torch.float).unsqueeze(0)

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
        return torch.ceil(seq_len.to(dtype=torch.float) / self.hop_length).to(
            dtype=torch.int)

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
                       win_length=self.win_length, pad_mode='reflect',
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

        # mask to zero any values beyond seq_len in batch, pad to multiple of `pad_to` (for efficiency)
        max_len = x.size(-1)
        mask = torch.arange(max_len).to(seq_len.dtype).to(x.device).expand(x.size(0),
                                                                           max_len) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)
        return x.to(dtype)

def dali_run(pipe, device):
    pipe.build()
    outs = pipe.run()
    return np.array(outs[0][0].as_cpu() if device == 'gpu' else outs[0][0])

def win_args(sample_rate, window_size_sec, window_stride_sec):
    win_length = int(sample_rate * window_size_sec)  # frame size
    hop_length = int(sample_rate * window_stride_sec)
    return win_length, hop_length

def torch_spectrogram(audio, sample_rate, device='cpu',
                      window_size=0.02, window_stride=0.01,
                      center=True, pad_mode='reflect',
                      window="hann", n_fft=None):
    audio = torch.tensor(audio, dtype=torch.float32)
    if device == 'gpu':
        audio = audio.cuda()
    win_length, hop_length = win_args(sample_rate, window_size, window_stride)
    n_fft = n_fft or 2 ** math.ceil(math.log2(win_length))
    window_fn = torch_windows.get(window, None)
    window_tensor = window_fn(win_length, periodic=False) if window_fn else None
    stft_out = torch.stft(audio, n_fft=n_fft, hop_length=hop_length,
                          win_length=win_length, pad_mode=pad_mode,
                          center=center, window=window_tensor.to(dtype=torch.float))
    # get power spectrum
    spectrogram = stft_out.pow(2).sum(-1)
    spectrogram = spectrogram.cpu().detach().numpy()
    return spectrogram

def librosa_spectrogram(audio_data, sample_rate, device='cpu',
                        window_size=0.02, window_stride=0.01,
                        center=True, pad_mode='reflect',
                        window="hann", n_fft=None):
    win_length, hop_length = win_args(sample_rate, window_size, window_stride)
    n_fft = n_fft or 2 ** math.ceil(math.log2(win_length))
    window_fn = torch_windows.get(window, None)
    window_tensor = window_fn(win_length, periodic=False).detach().numpy() if window_fn else None
    spectrogram = np.abs(
        librosa.stft(y=audio_data, n_fft=n_fft or win_length,
                     win_length=win_length, hop_length=hop_length, window=window_tensor))**2
    return spectrogram

def dali_spectrogram(audio_data, sample_rate, device='cpu',
                     window_size=0.02, window_stride=0.01,
                     center=True, pad_mode='reflect',
                     window="hann", n_fft=None):
    win_length, hop_length = win_args(sample_rate, window_size, window_stride)
    n_fft = n_fft or 2 ** math.ceil(math.log2(win_length))
    window_fn = torch_windows.get(window, None)
    window_tensor = window_fn(win_length, periodic=False).detach().numpy() if window_fn else None
    reflect_padding = 'reflect' == pad_mode
    @pipeline_def(batch_size=1, device_id=0, num_threads=3)
    def spectrogram_pipe():
        audio = fn.external_source(lambda: audio_data, device=device, batch=False)
        spectrogram = fn.spectrogram(audio, device=device, nfft=n_fft, reflect_padding=reflect_padding,
                                     center_windows=center, window_fn=window_tensor.tolist(),
                                     window_length=win_length, window_step=hop_length)
        return spectrogram
    return dali_run(spectrogram_pipe(), device=device)

def _testimpl_torch_vs_dali_spectrogram(device, pad_mode='reflect', center=True, atol=1e-03):
    for s in range(len(npy_files)):
        arr = _convert_samples_to_float32(np.load(npy_files[s]))
        torch_out = torch_spectrogram(arr, npy_files_sr, pad_mode=pad_mode, center=center, device=device)
        dali_out = dali_spectrogram(arr, npy_files_sr, pad_mode=pad_mode, center=center, device=device)
        rosa_out = librosa_spectrogram(arr, npy_files_sr, pad_mode=pad_mode, center=center, device=device)
        np.testing.assert_allclose(rosa_out, dali_out, atol=atol)
        np.testing.assert_allclose(torch_out, rosa_out, atol=atol)
        np.testing.assert_allclose(torch_out, dali_out, atol=atol)

def test_torch_vs_dali_spectrogram():
    for device in ['cpu', 'gpu']:
        yield _testimpl_torch_vs_dali_spectrogram, device

def torch_mel_fbank(spectrogram, sample_rate, device='cpu',
                    nfilt=64, lowfreq=0, highfreq=None):
    spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
    if device == 'gpu':
        spectrogram = spectrogram.cuda()
    n_fft = 2 * (spectrogram.shape[0] - 1)
    filterbanks = torch.tensor(
        librosa.filters.mel(sample_rate, n_fft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq), dtype=torch.float)
    if device == 'gpu':
        filterbanks = filterbanks.cuda()
    mel_spectrogram = torch.matmul(filterbanks.to(spectrogram.dtype), spectrogram)
    mel_spectrogram = mel_spectrogram.cpu().detach().numpy()
    return mel_spectrogram

def dali_mel_fbank(spectrogram_data, sample_rate, device='cpu',
                   nfilt=64, lowfreq=0, highfreq=None):
    @pipeline_def(batch_size=1, device_id=0, num_threads=3)
    def mel_fbank_pipe():
        spectrogram = fn.external_source(lambda: spectrogram_data, device=device, batch=False)
        mel_spectrogram = fn.mel_filter_bank(spectrogram, sample_rate=sample_rate, nfilter=nfilt,
                                             freq_low=lowfreq, freq_high=highfreq)
        return mel_spectrogram
    return dali_run(mel_fbank_pipe(), device=device)

def _testimpl_torch_vs_dali_mel_fbank(device):
    for s in range(len(npy_files)):
        arr = _convert_samples_to_float32(np.load(npy_files[s]))
        spec = torch_spectrogram(arr, npy_files_sr, device=device)
        torch_out = torch_mel_fbank(spec, npy_files_sr, device=device)
        dali_out = dali_mel_fbank(spec, npy_files_sr, device=device)
        np.testing.assert_allclose(torch_out, dali_out, atol=1e-04)

def test_torch_vs_dali_mel_fbank():
    for device in ['cpu', 'gpu']:
        yield _testimpl_torch_vs_dali_mel_fbank, device

def torch_log(x, device='cpu'):
    x = torch.tensor(x, dtype=torch.float32)
    if device == 'gpu':
        x = x.cuda()
    log_x = torch.log(x + 1e-20)
    log_x = log_x.cpu().detach().numpy()
    return log_x

def dali_log(x_data, device='cpu'):
    @pipeline_def(batch_size=1, device_id=0, num_threads=3)
    def log_pipe():
        x = fn.external_source(lambda: x_data, device=device, batch=False)
        log_x = fn.to_decibels(x, multiplier=np.log(10), reference=1.0, cutoff_db=-80)
        return log_x
    return dali_run(log_pipe(), device=device)

def _testimpl_torch_vs_dali_log(device):
    arr = _convert_samples_to_float32(np.load(npy_files[0]))
    spec = torch_spectrogram(arr, npy_files_sr, device=device)
    mel_spec = torch_mel_fbank(spec, npy_files_sr, device=device)
    torch_out = torch_log(mel_spec, device=device)
    dali_out = dali_log(mel_spec, device=device)
    np.testing.assert_allclose(torch_out, dali_out, atol=1e-5)

def torch_preemphasis(x, preemph, device='cpu'):
    x = torch.tensor(x, dtype=torch.float32)
    if device == 'gpu':
        x = x.cuda()
    y = torch.cat((x[0].unsqueeze(0), x[1:] - preemph * x[:-1]), dim=0)
    y = y.cpu().detach().numpy()
    return y

def dali_preemphasis(x_data, preemph, device='cpu'):
    @pipeline_def(batch_size=1, device_id=0, num_threads=3)
    def preemph_pipe():
        x = fn.external_source(lambda: x_data, device=device, batch=False)
        y = fn.preemphasis_filter(x, preemph_coeff=preemph)
        return y
    return dali_run(preemph_pipe(), device=device)

def _testimpl_torch_vs_dali_preemphasis(device):
    arr = _convert_samples_to_float32(np.load(npy_files[0]))
    torch_out = torch_preemphasis(arr, 0.97, device=device)
    dali_out = dali_preemphasis(arr, 0.97, device=device)
    # DALI and torch differ in the first element:
    # DALI: y[0] = x[0] - coeff * x[0]
    # Torch: y[0] = x[0]
    np.testing.assert_allclose(torch_out[1:], dali_out[1:], atol=1e-5)

def test_torch_vs_dali_preemphasis():
    for device in ['cpu', 'gpu']:
        yield _testimpl_torch_vs_dali_preemphasis, device

def torch_normalize(mel_spec, normalize_type, seq_len=None, device='cpu'):
    mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
    if seq_len is None:
        seq_len = torch.tensor(mel_spec.shape[2]).unsqueeze(0)
    if device == 'gpu':
        mel_spec = mel_spec.cuda()
    out = FilterbankFeatures().normalize_batch(
        mel_spec, seq_len, normalize_type=normalize_type)
    out = out.cpu().detach().numpy().squeeze(0)
    return out

def dali_normalize(mel_spec_data, normalize_type, device='cpu'):
    @pipeline_def(batch_size=1, device_id=0, num_threads=3)
    def log_pipe():
        data = fn.external_source(lambda: mel_spec_data, device=device, batch=False)
        if normalize_type == 'per_feature':
            out = fn.normalize(data, axes=[1], device=device, epsilon=4e-5, ddof=1)
        elif normalize_type == 'all_features':
            out = fn.normalize(data, axes=[0, 1], device=device, epsilon=4e-5, ddof=1)
        else:
            assert False
        return out
    return dali_run(log_pipe(), device=device)

def _testimpl_torch_vs_dali_normalize(normalize_type, device):
    for s in range(len(npy_files)):
        arr = _convert_samples_to_float32(np.load(npy_files[s]))
        spec = torch_spectrogram(arr, npy_files_sr, device=device)
        mel_spec = torch_mel_fbank(spec, npy_files_sr, device=device)
        log_features = torch_log(mel_spec, device=device)
        torch_out = torch_normalize(log_features, normalize_type=normalize_type, device=device)
        dali_out = dali_normalize(log_features, normalize_type=normalize_type, device=device)
        np.testing.assert_allclose(torch_out, dali_out, atol=1e-4)

def test_torch_vs_dali_normalize():
    for device in ['cpu', 'gpu']:
        for normalize_type in ['per_feature', 'all_features']:
            yield _testimpl_torch_vs_dali_normalize, normalize_type, device

@pipeline_def(batch_size=1, device_id=0, num_threads=3)
def rnnt_train_pipe(files, sample_rate=16000, silence_threshold=-80, preemph_coeff=.97,
                    window_size=.02, window_stride=.01, window="hann", nfeatures=64, nfft=512, frame_splicing=1,
                    lowfreq=0.0, highfreq=None, normalize_type='per_feature'):
    norm_axes = [1] if 'per_feature' else [0, 1]
    win_len, win_hop = win_args(sample_rate, window_size, window_stride)
    window_fn = torch_windows.get(window, None)
    window_fn_arg = window_fn(win_len, periodic=False).detach().numpy().tolist() if window_fn else None

    data, _ = fn.readers.file(files=files, device="cpu", random_shuffle=False, shard_id=0, num_shards=1)
    audio, _ = fn.decoders.audio(data, dtype=types.FLOAT, downmix=True)
    preemph_audio = fn.preemphasis_filter(audio, preemph_coeff=preemph_coeff, reflect_padding=False)
    spec = fn.spectrogram(preemph_audio, nfft=nfft, window_fn=window_fn_arg, window_length=win_len, window_step=win_hop,
                          center_windows=True, reflect_padding=True)

    mel_spec = fn.mel_filter_bank(spec, sample_rate=sample_rate, nfilter=nfeatures, freq_low=lowfreq, freq_high=highfreq)
    log_features = fn.to_decibels(mel_spec + 1e-20, multiplier=np.log(10), reference=1.0, cutoff_db=-80)

    if frame_splicing > 1:
        log_features = fn.transpose(log_features, perm=[1, 0])
        log_features = fn.reshape(log_features, rel_shape=[-1, frame_splicing])
        log_features = fn.pad(log_features, axes=[0], fill_value=0, align=frame_splicing, shape=[1])
        log_features = fn.transpose(log_features, perm=[1, 0])

    norm_log_features = fn.normalize(log_features, axes=[1], epsilon=4e-5, ddof=1)
    return norm_log_features, log_features, mel_spec, spec, preemph_audio, audio

# Test compares pre-calculated output of native data pipeline with an output
# from DALI data pipeline. There are few modification of native data pipeline
# comparing to the reference: random operations (i.e. dither and presampling
# aka "speed perturbation") are turned off
def test_rnnt_data_pipeline():
    preemph = 0.97
    n_fft = None
    sample_rate = npy_files_sr
    highfreq = None
    window_size = 0.02
    window_stride = 0.01
    normalize_type = 'per_feature'
    ref_pipeline = FilterbankFeatures(sample_rate=sample_rate, n_fft=n_fft, highfreq=highfreq, frame_splicing=1)
    recordings = []
    for fpath in npy_files:
        arr = np.load(fpath)
        arr = _convert_samples_to_float32(arr)
        recordings.append(arr)
    nrecordings = len(recordings)
    pipe = rnnt_train_pipe(audio_files, seed=42)
    pipe.build()
    reference_data = []
    for i in range(nrecordings):
        reference_data.append(
            ref_pipeline.forward(torch.tensor([recordings[i]]), torch.tensor([recordings[i].shape[0]]))
        )
    for i in range(nrecordings):
        dali_out = pipe.run()
        norm_log_features = np.array(dali_out[0][0])
        log_features = np.array(dali_out[1][0])
        mel_spec = np.array(dali_out[2][0])
        spec = np.array(dali_out[3][0])
        preemph_audio = np.array(dali_out[4][0])
        audio = np.array(dali_out[5][0])
        ref = reference_data[i].squeeze(0)
        assert ref.shape == norm_log_features.shape
        nfeatures, seq_len = ref.shape
        size = nfeatures * seq_len

        audio_ref = recordings[i]
        audio_len_ref = recordings[i].shape[0]
        np.testing.assert_allclose(audio, audio_ref, atol=1e-4)

        preemph_audio_ref = torch_preemphasis(audio_ref, preemph=preemph)
        np.testing.assert_allclose(preemph_audio, preemph_audio_ref, atol=1e-4)

        spec_ref = torch_spectrogram(preemph_audio_ref, npy_files_sr,
                                     window_size=window_size, window_stride=window_stride,
                                     center=True, pad_mode='reflect',
                                     window="hann", n_fft=n_fft)
        np.testing.assert_allclose(spec, spec_ref, atol=1e-4)

        mel_spec_ref = torch_mel_fbank(spec_ref, npy_files_sr)
        np.testing.assert_allclose(mel_spec, mel_spec_ref, atol=1e-4)

        log_features_ref = torch_log(mel_spec_ref)
        np.testing.assert_allclose(log_features, log_features_ref, atol=1e-3)
        log_features_ref2 = torch_log(mel_spec)
        np.testing.assert_allclose(log_features, log_features_ref2, atol=1e-4)

        norm_log_features_ref = torch_normalize(log_features_ref, normalize_type)
        np.testing.assert_allclose(norm_log_features, norm_log_features_ref, atol=1e-3)

        norm_log_features_ref2 = torch_normalize(log_features, normalize_type)
        np.testing.assert_allclose(norm_log_features, norm_log_features_ref2, atol=1e-4)

        # The reference pipeline calculate the number of windows in a wrong way, when using centered windows.
        # Here we are trying to recreate that behavior.
        seq_len = ref_pipeline.get_seq_len(torch.tensor([audio_len_ref]))
        norm_log_features_ref3 = torch_normalize(log_features_ref, normalize_type, seq_len=seq_len)
        ref_output = ref_pipeline.forward(torch.tensor([audio_ref]), torch.tensor([audio_len_ref])).squeeze(0)
        np.testing.assert_allclose(norm_log_features_ref3[:, :seq_len[0]], ref_output[:, :seq_len[0]], atol=1e-4)
