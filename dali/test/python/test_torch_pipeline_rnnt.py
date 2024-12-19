# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nose_utils import nottest
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def
import nvidia.dali.types as types
from test_utils import get_files, to_array
import numpy as np
import librosa
import torch
import math
import random
import os

# Filtering librispeech samples
audio_files = get_files("db/audio/wav", "wav")
audio_files = [file for file in audio_files if "237-134500" in file]

npy_files = [os.path.splitext(fpath)[0] + ".npy" for fpath in audio_files]
npy_files_sr = 16000

# From DeepLearningExamples


def _convert_samples_to_float32(samples):
    """Convert sample type to float32.
    Audio sample type is usually integer or float-point.
    Integers will be scaled to [-1, 1] in float32.
    """
    float32_samples = samples.astype("float32")
    if samples.dtype in np.sctypes["int"]:
        bits = np.iinfo(samples.dtype).bits
        float32_samples *= 1.0 / 2 ** (bits - 1)
    elif samples.dtype in np.sctypes["float"]:
        pass
    else:
        raise TypeError("Unsupported sample type: %s." % samples.dtype)
    return float32_samples


torch_windows = {
    "hann": torch.hann_window,
    "hamming": torch.hamming_window,
    "blackman": torch.blackman_window,
    "bartlett": torch.bartlett_window,
    "none": None,
}


def stack_subsample_frames(x, stacking=1, subsampling=1):
    """Stacks frames together across feature dim, and then subsamples

    input is batch_size, feature_dim, num_frames
    output is batch_size, feature_dim * stacking, num_frames / subsampling

    """
    seq = [x]
    for n in range(1, stacking):
        tmp = torch.zeros_like(x)
        tmp[:, :, :-n] = x[:, :, n:]
        seq.append(tmp)
    x = torch.cat(seq, dim=1)[:, :, ::subsampling]
    return x


class FilterbankFeatures:
    def __init__(
        self,
        sample_rate=16000,
        window_size=0.02,
        window_stride=0.01,
        window="hann",
        normalize="per_feature",
        n_fft=None,
        pad_amount=0,
        preemph=0.97,
        nfilt=64,
        lowfreq=0,
        highfreq=None,
        log=True,
        frame_splicing_stack=1,
        frame_splicing_subsample=1,
    ):
        self.win_length = int(sample_rate * window_size)
        self.hop_length = int(sample_rate * window_stride)
        self.n_fft = n_fft or 2 ** math.ceil(math.log2(self.win_length))

        self.normalize = normalize
        self.log = log
        self.frame_splicing_stack = frame_splicing_stack
        self.frame_splicing_subsample = frame_splicing_subsample
        self.nfilt = nfilt
        self.pad_amount = pad_amount
        self.preemph = preemph
        window_fn = torch_windows.get(window, None)
        self.window = window_fn(self.win_length, periodic=False) if window_fn else None
        filters = librosa.filters.mel(
            sr=sample_rate, n_fft=self.n_fft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq
        )
        self.fb = torch.tensor(filters, dtype=torch.float).unsqueeze(0)

    @staticmethod
    def normalize_batch(x, seq_len, normalize_type):
        constant = 1e-5
        if normalize_type == "per_feature":
            x_mean = torch.zeros((seq_len.shape[0], x.shape[1]), dtype=x.dtype, device=x.device)
            x_std = torch.zeros_like(x_mean)
            for i in range(x.shape[0]):
                x_mean[i, :] = x[i, :, : seq_len[i]].mean(dim=1)
                x_std[i, :] = x[i, :, : seq_len[i]].std(dim=1)
            # make sure x_std is not zero
            x_std += constant
            return (x - x_mean.unsqueeze(2)) / x_std.unsqueeze(2)
        elif normalize_type == "all_features":
            x_mean = torch.zeros(seq_len.shape, dtype=x.dtype, device=x.device)
            x_std = torch.zeros_like(x_mean)
            for i in range(x.shape[0]):
                x_mean[i] = x[i, :, : seq_len[i].item()].mean()
                x_std[i] = x[i, :, : seq_len[i].item()].std()
            # make sure x_std is not zero
            x_std += constant
            return (x - x_mean.view(-1, 1, 1)) / x_std.view(-1, 1, 1)
        else:
            return x

    def get_seq_len(self, seq_len):
        return seq_len.to(dtype=torch.int) // self.hop_length + 1

    def forward(self, inp, seq_len):
        x = inp
        dtype = x.dtype

        if self.pad_amount > 0:
            x = torch.nn.functional.pad(
                x.unsqueeze(1), (self.pad_amount, self.pad_amount), "reflect"
            ).squeeze(1)
            seq_len = seq_len + 2 * self.pad_amount

        seq_len = self.get_seq_len(seq_len)

        # do preemphasis
        if self.preemph is not None:
            x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - self.preemph * x[:, :-1]), dim=1)

        # do stft
        x = torch.stft(
            x,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            pad_mode="reflect",
            center=True,
            window=self.window.to(dtype=torch.float).to(x.device),
            return_complex=True,
        )
        x = torch.view_as_real(x)

        # get power spectrum
        x = x.pow(2).sum(-1)

        # dot with filterbank energies
        x = torch.matmul(self.fb.to(x.dtype), x)

        # log features if required
        if self.log:
            x = torch.log(x + 1e-20)

        # frame splicing if required
        if self.frame_splicing_stack > 1 or self.frame_splicing_subsample:
            x = stack_subsample_frames(
                x, stacking=self.frame_splicing_stack, subsampling=self.frame_splicing_subsample
            )

        # normalize if required
        if self.normalize:
            x = self.normalize_batch(x, seq_len, normalize_type=self.normalize)

        # mask to zero any values beyond seq_len in batch,
        # pad to multiple of `pad_to` (for efficiency)
        max_len = x.size(-1)
        seq = torch.arange(max_len).to(seq_len.dtype).to(x.device)
        mask = seq.expand(x.size(0), max_len) >= seq_len.unsqueeze(1)
        x = x.masked_fill(mask.unsqueeze(1).to(device=x.device), 0)
        return x.to(dtype)


def dali_run(pipe, device):
    outs = pipe.run()
    return to_array(outs[0])[0]


def win_args(sample_rate, window_size_sec, window_stride_sec):
    win_length = int(sample_rate * window_size_sec)  # frame size
    hop_length = int(sample_rate * window_stride_sec)
    return win_length, hop_length


def torch_spectrogram(
    audio,
    sample_rate,
    device="cpu",
    window_size=0.02,
    window_stride=0.01,
    center=True,
    pad_mode="reflect",
    window="hann",
    n_fft=None,
):
    audio = torch.tensor(audio, dtype=torch.float32)
    if device == "gpu":
        audio = audio.cuda()
    win_length, hop_length = win_args(sample_rate, window_size, window_stride)
    n_fft = n_fft or 2 ** math.ceil(math.log2(win_length))
    window_fn = torch_windows.get(window, None)
    window_tensor = window_fn(win_length, periodic=False) if window_fn else None
    stft_out = torch.stft(
        audio,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_length,
        pad_mode=pad_mode,
        center=center,
        window=window_tensor.to(dtype=torch.float),
        return_complex=True,
    )
    stft_out = torch.view_as_real(stft_out)
    # get power spectrum
    spectrogram = stft_out.pow(2).sum(-1)
    spectrogram = spectrogram.cpu().numpy()
    return spectrogram


def torch_mel_fbank(spectrogram, sample_rate, device="cpu", nfilt=64, lowfreq=0, highfreq=None):
    spectrogram = torch.tensor(spectrogram, dtype=torch.float32)
    if device == "gpu":
        spectrogram = spectrogram.cuda()
    n_fft = 2 * (spectrogram.shape[0] - 1)
    filterbanks = torch.tensor(
        librosa.filters.mel(sr=sample_rate, n_fft=n_fft, n_mels=nfilt, fmin=lowfreq, fmax=highfreq),
        dtype=torch.float,
    )
    if device == "gpu":
        filterbanks = filterbanks.cuda()
    mel_spectrogram = torch.matmul(filterbanks.to(spectrogram.dtype), spectrogram)
    mel_spectrogram = mel_spectrogram.cpu().numpy()
    return mel_spectrogram


def torch_log(x, device="cpu"):
    x = torch.tensor(x, dtype=torch.float32)
    if device == "gpu":
        x = x.cuda()
    log_x = torch.log(x + 1e-20)
    log_x = log_x.cpu().numpy()
    return log_x


def torch_preemphasis(x, preemph, device="cpu"):
    x = torch.tensor(x, dtype=torch.float32)
    if device == "gpu":
        x = x.cuda()
    y = torch.cat((x[0].unsqueeze(0), x[1:] - preemph * x[:-1]), dim=0)
    y = y.cpu().numpy()
    return y


def torch_normalize(mel_spec, normalize_type, seq_len=None, device="cpu"):
    mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
    if seq_len is None:
        seq_len = torch.tensor(mel_spec.shape[2]).unsqueeze(0)
    if device == "gpu":
        mel_spec = mel_spec.cuda()
    out = FilterbankFeatures().normalize_batch(mel_spec, seq_len, normalize_type=normalize_type)
    out = out.cpu().numpy().squeeze(0)
    return out


def torch_frame_splicing(mel_spec, stacking=1, subsampling=1, device="cpu"):
    mel_spec = torch.tensor(mel_spec, dtype=torch.float32).unsqueeze(0)
    if device == "gpu":
        mel_spec = mel_spec.cuda()
    out = stack_subsample_frames(mel_spec, stacking=stacking, subsampling=subsampling)
    out = out.cpu().numpy().squeeze(0)
    return out


def dali_frame_splicing_graph(x, nfeatures, x_len, stacking=1, subsampling=1):
    if stacking > 1:
        seq = [x]
        for n in range(1, stacking):
            f = fn.slice(x, n, x_len, axes=(1,), out_of_bounds_policy="pad", fill_values=0)
            seq.append(f)
        x = fn.cat(*seq, axis=0)
        nfeatures = nfeatures * stacking
    if subsampling > 1:
        out_len = (x_len + subsampling - 1) // subsampling
        m = fn.transforms.scale(scale=[subsampling, 1], center=[0.5, 0])
        x = fn.reshape(x, rel_shape=[1, 1, -1], layout="HWC")  # Layout required by WarpAffine
        size = fn.stack(nfeatures, out_len)
        x = fn.warp_affine(x, matrix=m, size=size, interp_type=types.INTERP_NN)
        x = fn.reshape(x, rel_shape=[1, 1], layout="ft")
    return x


def torch_reflect_pad(x, pad_amount, device="cpu"):
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
    if device == "gpu":
        x = x.cuda()
    x = torch.nn.functional.pad(x.unsqueeze(1), (pad_amount, pad_amount), "reflect").squeeze(1)
    x = x.cpu().numpy().squeeze(0)
    return x


def dali_reflect_pad_graph(x, pad_amount):
    pad_start = x[pad_amount:0:-1]
    pad_end = x[-2 : -pad_amount - 2 : -1]
    x = fn.cat(pad_start, x, pad_end, axis=0)
    return x


@pipeline_def(batch_size=1, device_id=0, num_threads=4, exec_dynamic=True)
def rnnt_train_pipe(
    files,
    sample_rate,
    pad_amount=0,
    preemph_coeff=0.97,
    window_size=0.02,
    window_stride=0.01,
    window="hann",
    nfeatures=64,
    nfft=512,
    frame_splicing_stack=1,
    frame_splicing_subsample=1,
    lowfreq=0.0,
    highfreq=None,
    normalize_type="per_feature",
    speed_perturb=False,
    silence_trim=False,
    device="cpu",
):
    assert normalize_type == "per_feature" or normalize_type == "all_features"
    norm_axes = [1] if normalize_type == "per_feature" else [0, 1]
    win_len, win_hop = win_args(sample_rate, window_size, window_stride)
    window_fn = torch_windows.get(window, None)
    window_fn_arg = window_fn(win_len, periodic=False).numpy().tolist() if window_fn else None

    data, _ = fn.readers.file(files=files, device="cpu", random_shuffle=False)
    audio, _ = fn.decoders.audio(data, dtype=types.FLOAT, downmix=True)

    # splicing with subsampling doesn't work if audio_len is a GPU data node
    if device == "gpu":
        audio = audio.gpu()

    # Speed perturbation 0.85x - 1.15x
    if speed_perturb:
        target_sr_factor = fn.random.uniform(device="cpu", range=(1 / 1.15, 1 / 0.85))
        audio = fn.audio_resample(audio, scale=target_sr_factor)

    # Silence trimming
    if silence_trim:
        begin, length = fn.nonsilent_region(audio, cutoff_db=-80)
        audio = audio[begin : begin + length]

    audio_shape = audio.shape(dtype=types.INT32)
    orig_audio_len = audio_shape[0]

    if pad_amount > 0:
        audio_len = orig_audio_len + 2 * pad_amount
        padded_audio = dali_reflect_pad_graph(audio, pad_amount)
    else:
        audio_len = orig_audio_len
        padded_audio = audio

    # Preemphasis filter
    preemph_audio = fn.preemphasis_filter(padded_audio, preemph_coeff=preemph_coeff, border="zero")

    # Spectrogram
    spec = fn.spectrogram(
        preemph_audio,
        nfft=nfft,
        window_fn=window_fn_arg,
        window_length=win_len,
        window_step=win_hop,
        center_windows=True,
        reflect_padding=True,
    )
    # Mel spectrogram
    mel_spec = fn.mel_filter_bank(
        spec, sample_rate=sample_rate, nfilter=nfeatures, freq_low=lowfreq, freq_high=highfreq
    )

    # Log
    log_features = fn.to_decibels(
        mel_spec + 1e-20, multiplier=np.log(10), reference=1.0, cutoff_db=-80
    )

    # Frame splicing
    if frame_splicing_stack > 1 or frame_splicing_subsample > 1:
        spec_len = audio_len // win_hop + 1
        log_features_spliced = dali_frame_splicing_graph(
            log_features,
            nfeatures,
            spec_len,
            stacking=frame_splicing_stack,
            subsampling=frame_splicing_subsample,
        )
    else:
        log_features_spliced = log_features

    # Normalization
    if normalize_type:
        norm_log_features = fn.normalize(
            log_features_spliced, axes=norm_axes, device=device, epsilon=4e-5, ddof=1
        )
    else:
        norm_log_features = log_features_spliced

    return (
        norm_log_features,
        log_features_spliced,
        log_features,
        mel_spec,
        spec,
        preemph_audio,
        padded_audio,
        audio,
    )


recordings = []
for fpath in npy_files:
    arr = np.load(fpath)
    arr = _convert_samples_to_float32(arr)
    recordings.append(arr)
nrecordings = len(recordings)

# Test compares pre-calculated output of native data pipeline with an output
# from DALI data pipeline. There are few modification of native data pipeline
# comparing to the reference: random operations (i.e. dither and presampling
# aka "speed perturbation") are turned off


def _testimpl_rnnt_data_pipeline(
    device,
    pad_amount=0,
    preemph_coeff=0.97,
    window_size=0.02,
    window_stride=0.01,
    window="hann",
    nfeatures=64,
    n_fft=512,
    frame_splicing_stack=1,
    frame_splicing_subsample=1,
    lowfreq=0.0,
    highfreq=None,
    normalize_type="per_feature",
    batch_size=32,
):
    sample_rate = npy_files_sr
    speed_perturb = False
    silence_trim = False

    ref_pipeline = FilterbankFeatures(
        sample_rate=sample_rate,
        window_size=window_size,
        window_stride=window_stride,
        window=window,
        normalize=normalize_type,
        n_fft=n_fft,
        pad_amount=pad_amount,
        preemph=preemph_coeff,
        nfilt=nfeatures,
        lowfreq=lowfreq,
        highfreq=highfreq,
        log=True,
        frame_splicing_stack=frame_splicing_stack,
        frame_splicing_subsample=frame_splicing_subsample,
    )
    reference_data = []
    for i in range(nrecordings):
        reference_data.append(
            ref_pipeline.forward(
                torch.tensor([recordings[i]]), torch.tensor([recordings[i].shape[0]])
            )
        )

    pipe = rnnt_train_pipe(
        audio_files,
        sample_rate,
        pad_amount,
        preemph_coeff,
        window_size,
        window_stride,
        window,
        nfeatures,
        n_fft,
        frame_splicing_stack,
        frame_splicing_subsample,
        lowfreq,
        highfreq,
        normalize_type,
        speed_perturb,
        silence_trim,
        device,
        seed=42,
        batch_size=batch_size,
    )
    nbatches = (nrecordings + batch_size - 1) // batch_size
    i = 0
    for b in range(nbatches):
        dali_out = list(pipe.run())
        for s in range(batch_size):
            if i >= nrecordings:
                break

            (
                norm_log_features,
                log_features_spliced,
                log_features,
                mel_spec,
                spec,
                preemph_audio,
                padded_audio,
                audio,
            ) = [to_array(out[s]) for out in dali_out]

            ref = np.array(reference_data[i].squeeze(0))
            assert ref.shape == norm_log_features.shape, f"{ref.shape}, {norm_log_features.shape}"
            nfeatures, seq_len = ref.shape

            audio_ref = recordings[i]
            np.testing.assert_allclose(audio, audio_ref, atol=1e-4)

            padded_audio_ref = torch_reflect_pad(audio, pad_amount)
            np.testing.assert_equal(padded_audio, padded_audio_ref)

            preemph_audio_ref = torch_preemphasis(padded_audio_ref, preemph=preemph_coeff)
            np.testing.assert_allclose(preemph_audio, preemph_audio_ref, atol=1e-4)

            spec_ref = torch_spectrogram(
                preemph_audio_ref,
                npy_files_sr,
                window_size=window_size,
                window_stride=window_stride,
                center=True,
                pad_mode="reflect",
                window=window,
                n_fft=n_fft,
            )
            np.testing.assert_allclose(spec, spec_ref, atol=1e-4)

            mel_spec_ref = torch_mel_fbank(spec_ref, npy_files_sr)
            np.testing.assert_allclose(mel_spec, mel_spec_ref, atol=1e-4)

            log_features_ref = torch_log(mel_spec_ref)
            np.testing.assert_allclose(log_features, log_features_ref, atol=1e-3)
            log_features_ref2 = torch_log(mel_spec)
            np.testing.assert_allclose(log_features, log_features_ref2, atol=1e-4)

            log_features_spliced_ref = torch_frame_splicing(
                log_features_ref,
                stacking=frame_splicing_stack,
                subsampling=frame_splicing_subsample,
            )
            np.testing.assert_allclose(log_features_spliced, log_features_spliced_ref, atol=1e-3)

            log_features_spliced_ref2 = torch_frame_splicing(
                log_features, stacking=frame_splicing_stack, subsampling=frame_splicing_subsample
            )
            np.testing.assert_allclose(log_features_spliced, log_features_spliced_ref2, atol=1e-4)

            norm_log_features_ref = torch_normalize(log_features_spliced_ref, normalize_type)
            np.testing.assert_allclose(norm_log_features, norm_log_features_ref, atol=1e-3)

            norm_log_features_ref2 = torch_normalize(log_features_spliced, normalize_type)
            np.testing.assert_allclose(norm_log_features, norm_log_features_ref2, atol=1e-4)

            # Full pipeline
            np.testing.assert_allclose(norm_log_features, ref, atol=1e-3)

            i += 1


def test_rnnt_data_pipeline():
    preemph_coeff = 0.97
    window_size = 0.02
    window_stride = 0.01
    window = "hann"
    nfeatures = 64
    n_fft = 512
    lowfreq = 0.0
    highfreq = None
    for device in ["cpu", "gpu"]:
        for frame_splicing_stack, frame_splicing_subsample in [(1, 1), (3, 2)]:
            for normalize_type in ["per_feature", "all_features"]:
                pad_amount = random.choice([0, 16])
                yield (
                    _testimpl_rnnt_data_pipeline,
                    device,
                    pad_amount,
                    preemph_coeff,
                    window_size,
                    window_stride,
                    window,
                    nfeatures,
                    n_fft,
                    frame_splicing_stack,
                    frame_splicing_subsample,
                    lowfreq,
                    highfreq,
                    normalize_type,
                )


@nottest  # To be run manually to check perf
def test_rnnt_data_pipeline_throughput(
    pad_amount=0,
    preemph_coeff=0.97,
    window_size=0.02,
    window_stride=0.01,
    window="hann",
    nfeatures=64,
    n_fft=512,
    frame_splicing_stack=1,
    frame_splicing_subsample=1,
    speed_perturb=True,
    silence_trim=True,
    lowfreq=0.0,
    highfreq=None,
    normalize_type="per_feature",
    batch_size=32,
):
    sample_rate = npy_files_sr
    device = "gpu"
    pipe = rnnt_train_pipe(
        audio_files,
        sample_rate,
        pad_amount,
        preemph_coeff,
        window_size,
        window_stride,
        window,
        nfeatures,
        n_fft,
        frame_splicing_stack,
        frame_splicing_subsample,
        lowfreq,
        highfreq,
        normalize_type,
        speed_perturb,
        silence_trim,
        device,
        seed=42,
        batch_size=batch_size,
    )

    import time
    from test_utils import AverageMeter

    end = time.time()
    data_time = AverageMeter()
    iters = 1000
    for j in range(iters):
        pipe.run()
        data_time.update(time.time() - end)
        if j % 100 == 0:
            print(
                f"run {j+1}/ {iters}, avg time: {data_time.avg} [s], "
                f"worst time: {data_time.max_val} [s], "
                f"speed: {batch_size / data_time.avg} [recordings/s]"
            )
        end = time.time()
