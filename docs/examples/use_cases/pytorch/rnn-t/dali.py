import nvidia.dali
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np

def print_dict(d):
    maxLen = max([len(ii) for ii in d.keys()])
    fmtString = '\t%' + str(maxLen) + 's : %s'
    print('Arguments:')
    for keyPair in sorted(d.items()):
        print(fmtString % keyPair)

def gen_filelist(json_path):
    import json, os
    with open(json_path) as f:
        librispeech_json = json.load(f)
    output_files = {}
    transcripts = {}
    curr_label = 0
    for original_sample in librispeech_json:
        transcripts[curr_label] = original_sample['transcript']
        for file in original_sample['files']:
            output_files[os.path.join(file['fname'])] = curr_label
        curr_label += 1
    return output_files, transcripts


def dict_to_file(dict, filename):
    with open(filename, "w") as f:
        for key, value in dict.items():
            f.write("{} {}\n".format(key, value))


class DaliTrainPipeline(nvidia.dali.pipeline.Pipeline):
    def __init__(self,
                 device_id,
                 n_devices,
                 file_root,
                 file_list,
                 batch_size,
                 sample_rate,
                 resample_range,
                 window_size,
                 window_stride,
                 nfeatures,
                 nfft,
                 frame_splicing_factor,
                 dither_coeff,
                 silence_threshold,
                 preemph_coeff,
                 lowfreq=0.0,
                 highfreq=0.0,
                 num_threads=1, exec_async=False, exec_pipelined=False):
        """

        Args:
            nfft: Size of the FFT
            window_size: Window size as fraction of the sample rate
            window_stride: Window step as fraction of the sample rate
            dither_coeff:
            preemph_coeff:
            batch_size:
            num_threads:
            device_id:
            silence_threshold: dBs
        """
        super(DaliTrainPipeline, self).__init__(batch_size, num_threads, device_id,
                                                exec_async=exec_async,
                                                exec_pipelined=exec_pipelined, seed=42)

        self.sample_rate = sample_rate
        self.dither = dither_coeff
        self.frame_splicing_factor = frame_splicing_factor

        self.speed_perturbation_coeffs = ops.Uniform(device="cpu", range=resample_range)

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
        self.splicing_pad = ops.Pad(axes=[0], fill_value=0, align=frame_splicing_factor,
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
        resample_coeffs = self.speed_perturbation_coeffs()
        audio, sr = self.decode(input, sample_rate=resample_coeffs * self.sample_rate)

        audio = self.remove_silence(audio)

        audio = audio + self.normal_distribution(audio) * self.dither

        # DALI's preemph work a little bit different than the one in native code.
        # The difference occurs in first value in buffer.
        audio = self.preemph(audio)

        audio = self.spectrogram(audio)
        audio = self.mel_fbank(audio)
        audio = self.log_features(audio)
        audio, audio_sh = self._splice_frames(audio)

        # This normalization goes across ax=0, since
        # the frame splicing didn't transpose the tensor back
        audio = self.normalize(audio)

        return audio.gpu(), label.gpu(), audio_sh.gpu()


class DaliRnntIterator(object):
    """
    Returns batches of data for RNNT training:
    audio_signal, audio_signal_length, transcript, transcript_length

    dali_pipelines: List of DALI pipelines. One per every GPU.

    TODO: describe
    You rather won't need to create it yourself. Use DataLayer instead
    """

    def __init__(self, dali_pipelines, transcripts):
        self.transcripts = transcripts
        from nvidia.dali.plugin.pytorch import DALIGenericIterator
        # When modifying DALI pipeline, make sure you update `output_map` in DALIGenericIterator invocation below
        self.dali_it = DALIGenericIterator(dali_pipelines, ["audio", "label", "audio_shape"],
                                           len(transcripts), dynamic_shape=True)

    @staticmethod
    def _str2list(s):
        """
        Returns list of floats, that represents given string.
        '0.' denotes separator
        '1.' denotes 'a'
        Assumes, that the string is lower case.
        """
        return [max(0, ord(c) - 96.) for c in s]

    @staticmethod
    def _pad_lists(lists, pad_val=0):
        """
        Pads lists, so that all have the same size.
        Returns list with actual sizes of corresponding input lists
        """
        max_length = 0
        sizes = []
        for l in lists:
            sizes.append(len(l))
            max_length = max_length if len(l) < max_length else len(l)
        for l in lists:
            l += [pad_val] * (max_length - len(l))
        return sizes

    def _gen_transcripts(self, labels):
        """
        Generate transcripts in format expected by NN
        """
        import torch
        lists = [self._str2list(self.transcripts[lab.item()]) for lab in labels]
        sizes = self._pad_lists(lists)
        return torch.tensor(lists).cuda(), torch.tensor(sizes, dtype=torch.int32).cuda()

    def __next__(self):
        data = self.dali_it.__next__()
        transcripts, transcripts_lengths = self._gen_transcripts(data[0]["label"])
        return data[0]["audio"], data[0]["audio_shape"][:, 0], transcripts, transcripts_lengths

    def next(self):
        return self.__next__()

    def __iter__(self):
        return self


class DaliDataLayer:
    """
    DataLayer is the main entry point to the data preprocessing pipeline.
    To use, create an object and then just iterate over `data_iterator`.
    DataLayer will do the rest for you.
    Example:
        data_layer = DaliDataLayer(path, json, bs)
        data_it = data_layer.data_iterator
        for data in data_it:
            print(data)  # Here's your preprocessed data
    """

    def __init__(self,
                 dataset_path,
                 json_name,
                 batch_size,
                 ngpus,
                 nfeatures=64,
                 sample_rate=16000,
                 nfft=512,
                 window_size=.02,
                 window_stride=.01,
                 dither_coeff=.00001,
                 frame_splicing_factor=3,
                 resample_range=[.9, 1.1],
                 silence_threshold=-60,
                 preemph_coeff=.97):
        self._dali_data_iterator = self._init_iterator(
            ngpus=ngpus, dataset_path=dataset_path, json_name=json_name, batch_size=batch_size,
            nfeatures=nfeatures, sample_rate=sample_rate, nfft=nfft, window_size=window_size,
            window_stride=window_stride, dither_coeff=dither_coeff,
            frame_splicing_factor=frame_splicing_factor, resample_range=resample_range,
            silence_threshold=silence_threshold, preemph_coeff=preemph_coeff)

    @classmethod
    def _init_iterator(self, **kwargs):
        """
        :return: data iterator. The data will be preprocessed within DALI.
        """
        import os
        print("\nInitializing DALI layer")
        print_dict(kwargs)
        print("\n")
        self.file_list_path = "/tmp/file_list.txt"
        jn = kwargs["json_name"]
        dp = kwargs["dataset_path"]
        output_files, transcripts = gen_filelist(jn if jn[0] == '/' else os.path.join(dp, jn))
        dict_to_file(output_files, self.file_list_path)
        self.dataset_size = len(output_files)

        dali_pipelines = [
            DaliTrainPipeline(
                device_id=idx,
                file_list=self.file_list_path,
                n_devices=kwargs["ngpus"],
                file_root=kwargs["dataset_path"],
                batch_size=kwargs["batch_size"],
                nfeatures=kwargs["nfeatures"],
                resample_range=kwargs["resample_range"],
                sample_rate=kwargs["sample_rate"],
                nfft=kwargs["nfft"],
                window_size=kwargs["window_size"],
                window_stride=kwargs["window_stride"],
                dither_coeff=kwargs["dither_coeff"],
                frame_splicing_factor=kwargs["frame_splicing_factor"],
                silence_threshold=kwargs["silence_threshold"],
                preemph_coeff=kwargs["preemph_coeff"])
            for idx in range(kwargs["ngpus"])
        ]

        return DaliRnntIterator(dali_pipelines, transcripts)

    def __len__(self):
        return self.dataset_size

    @property
    def data_iterator(self):
        return self._dali_data_iterator


# if __name__ == '__main__':
#     dali_data_layer = DaliDataLayer(dataset_path="/storage/LibriSpeech/",
#                                     json_name="librispeech-train-clean-100-wav.json", batch_size=2,
#                                     ngpus=1)
#     for data in dali_data_layer.data_iterator:
#         print([d.shape for d in data])
#         print(data)
#         break

def test_dali():
    pipe=DaliTrainPipeline(
        0,1,
        None,
        None,
        sample_rate=16000,
        nfft=512,
        nfeatures=64,
        window_size=0.02,
        window_stride=0.01,
        dither_coeff=0.00001,
        batch_size=2,
        lowfreq=0.0,
        frame_splicing_factor=3,
        resample_range=[.9, 1.1],
        silence_threshold=-60,
        highfreq=0.0,
        preemph_coeff=0.97,
        num_threads=1)
    pipe.build()
    oo=pipe.run()
    import ipdb; ipdb.set_trace()
    pass

if __name__ == '__main__':
    test_dali()