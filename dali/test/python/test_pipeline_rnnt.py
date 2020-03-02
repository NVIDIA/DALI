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
import numpy as np
import os
import test_utils

dali_extra_path = test_utils.get_dali_extra_path()


class RnntTrainPipeline(nvidia.dali.pipeline.Pipeline):
    def __init__(self,
                 device_id,
                 n_devices,
                 file_root,
                 file_list,
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
                 num_threads=1, exec_async=True, exec_pipelined=True):
        super().__init__(batch_size, num_threads, device_id,
                         exec_async=exec_async,
                         exec_pipelined=exec_pipelined, seed=42)

        self.dither = dither
        self.sample_rate = sample_rate
        self.frame_splicing_factor = frame_splicing_factor

        self.read = ops.FileReader(file_root=file_root, file_list=file_list, device="cpu",
                                   shard_id=device_id, num_shards=n_devices)

        self.decode = ops.AudioDecoder(device="cpu", dtype=types.FLOAT, downmix=True)

        self.normal_distribution = ops.NormalDistribution(device="cpu")

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
        self.trim_silence = ops.Slice(device="cpu", normalized_anchor=False, normalized_shape=False,
                                      axes=[0], image_type=types.ANY_DATA)
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
        out = self.trim_silence(input, self.to_float(begin), self.to_float(len))
        return out

    def define_graph(self):
        input, label = self.read()
        # No resampling (aka "speed perturbation"), because of randomness of this operation
        audio, sr = self.decode(input)

        audio = self.remove_silence(audio)

        audio = audio + self.normal_distribution(audio) * self.dither

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

        return audio.gpu(), audio_sh.gpu()


def test_rnnt_data_pipeline():
    """
    Test compares pre-calculated output of native data pipeline with an output
    from DALI data pipeline. There are few modification of native data pipeline
    comparing to the reference:
    1. Presampling (aka "speed perturbation") is turned off
    2. Since DALI, as an optimization, doesn't perform one transposition in frame splicing,
       the result from DALI pipeline has to be transposed to fit the reference data.
    """
    batch_size = 2
    data_path = os.path.join(dali_extra_path, "db", "audio", "rnnt_data_pipeline")
    ref_names = ["and_showed_itself_output.npy", "asked_her_father_output.npy"]
    reference_data = [np.load(os.path.join(data_path, ref))[0] for ref in ref_names]
    pipe = RnntTrainPipeline(device_id=0, n_devices=1, file_root=data_path,
                             file_list=os.path.join(data_path, "file_list.txt"),
                             batch_size=batch_size)
    pipe.build()
    dali_out = pipe.run()
    for sample_idx in range(batch_size):
        output_data = dali_out[0].as_cpu().at(sample_idx)
        output_data = np.transpose(output_data, (1, 0))
        audio_len = dali_out[1].as_cpu().at(sample_idx)[0]
        assert audio_len == reference_data[sample_idx].shape[1]
        assert reference_data[sample_idx].shape == output_data.shape
        size = reference_data[sample_idx].flatten().shape[0]
        assert np.sum(
            np.isclose(reference_data[sample_idx], output_data, atol=.01, rtol=0)) / size > .98
