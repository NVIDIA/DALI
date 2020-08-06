from nvidia.dali.pipeline import Pipeline
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np
import math
import os.path
import PIL.Image
from test_utils import check_batch

def init_video_data():
    batch_size = 2
    video_directory = os.path.join(os.environ['DALI_EXTRA_PATH'], "db", "video", "sintel", "video_files")

    video_files=[os.path.join(video_directory, f) for f in os.listdir(video_directory)]
    print(video_files)

    video_pipe = dali.pipeline.Pipeline(batch_size, 3, 0, seed=16)
    with video_pipe:
        input = fn.video_reader(device="gpu", filenames=video_files, sequence_length=32, stride=5)
        video_pipe.set_outputs(input)

    video_pipe.build()
    out = video_pipe.run()
    in_seq = out[0].as_cpu().at(0)
    return in_seq

frames_fhwc = init_video_data()
frames_fchw = frames_fhwc.transpose([0, 3, 1, 2])

# gets overlapping sequences, starting at iteration number
def GetSequences(channel_first, length, batch_size):
    source = frames_fchw if channel_first else frames_fhwc
    N = source.shape[0]
    def get_seq(id):
        ret = []
        for k in range(length):
            i = (id + k) % N
            ret.append(source[i])
        return np.array(ret)
    def get_batch(iter):
        return [get_seq(iter * batch_size + i) for i in range(batch_size)]
    return get_batch

resample_dali2pil = {
    types.INTERP_NN         : PIL.Image.NEAREST,
    types.INTERP_TRIANGULAR : PIL.Image.BILINEAR,
    types.INTERP_CUBIC      : PIL.Image.BICUBIC,
    types.INTERP_LANCZOS3   : PIL.Image.LANCZOS
}

def resize_PIL(channel_first, interp, w, h):
    pil_resample = resample_dali2pil[interp]
    def resize(input):
        num_frames = input.shape[0]
        out_seq = []
        for i in range(num_frames):
            frame = input[i]
            if channel_first:
                frame = frame.transpose([1,2,0])
            out_frame = PIL.Image.fromarray(frame).resize([w, h], resample=pil_resample)
            out_frame = np.array(out_frame)
            if channel_first:
                out_frame = out_frame.transpose([2,0,1])
            out_seq.append(out_frame)
        return np.array(out_seq)
    return resize

def create_ref_pipe(channel_first, seq_len, interp, w, h, batch_size = 2):
    pipe = dali.pipeline.Pipeline(batch_size,1,0,0, exec_async=False, exec_pipelined=False)
    with pipe:
        layout = "FCHW" if channel_first else "FHWC"
        ext = fn.external_source(GetSequences(channel_first, seq_len, batch_size), layout = layout)
        pil_resized = fn.python_function(ext, function=resize_PIL(channel_first, interp, w, h), batch_processing = False)
        pil_resized = fn.reshape(pil_resized, layout=layout)
        pipe.set_outputs(pil_resized)
    return pipe

def create_dali_pipe(channel_first, seq_len, interp, w, h, batch_size = 2):
    pipe = dali.pipeline.Pipeline(batch_size,1,0,0)
    with pipe:
        layout = "FCHW" if channel_first else "FHWC"
        ext = fn.external_source(GetSequences(channel_first, seq_len, batch_size), layout = layout)
        dali_resized_cpu = fn.resize(ext,       resize_x = w, resize_y = h, interp_type = interp)
        dali_resized_gpu = fn.resize(ext.gpu(), resize_x = w, resize_y = h, interp_type = interp, minibatch_size=4)
        pipe.set_outputs(dali_resized_cpu, dali_resized_gpu)
    return pipe

def _test_resize(layout, interp, w, h):
    channel_first = (layout == "FCHW")
    pipe_dali = create_dali_pipe(channel_first, 8, interp, w, h)
    pipe_dali.build()
    pipe_ref = create_ref_pipe(channel_first, 8, interp, w, h)
    pipe_ref.build()
    eps = 1e-2
    max_err = 5
    for iter in range(4):
        out_dali = pipe_dali.run()
        out_ref = pipe_ref.run()
        check_batch(out_dali[0], out_ref[0], 2, eps=eps, max_allowed_error=max_err)
        check_batch(out_dali[1], out_ref[0], 2, eps=eps, max_allowed_error=max_err)

def test_resize():
    for layout in ["FHWC", "FCHW"]:
        for interp, w, h in [(types.INTERP_NN, 640, 480),
            (types.INTERP_TRIANGULAR, 100, 80),
            (types.INTERP_LANCZOS3, 200, 100)]:
            yield _test_resize, layout, interp, w, h
