import os
import numpy as np
from functools import partial

import nvidia.dali as dali
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

from PIL import Image as Image


import os
import numpy as np

import nvidia.dali as dali
import nvidia.dali.ops as ops
import nvidia.dali.types as types

from PIL import Image as Image


video_directory = os.path.join(
    os.environ['DALI_EXTRA_PATH'], "db", "video", "sintel", "video_files")
# video_directory = '/tmp/video_resolution/'

video_files = [video_directory + '/' + f for f in os.listdir(video_directory)]


def save_frame(frame, sample_id, frame_id, msg):
    if type(frame) != np.ndarray:
        frame = frame.as_array()[0]
    frame_image = Image.fromarray(frame)
    frame_image.save(
        '/home/awolant/Projects/test_video/{}_{}_{}.jpeg'.format(msg, sample_id, frame_id))


pipeline_params = {
    'num_threads': 8,
    'device_id': 0,
    'seed': 0
}

video_reader_params = {
    'device': 'gpu',
    'filenames': video_files,
    'sequence_length': 32,
    'random_shuffle': False
}

resize_params = {
    'resize_x': 300,
    'resize_y': 200,
    'interp_type': types.DALIInterpType.INTERP_CUBIC,
    'minibatch_size': 8
}


def video_reader_pipeline_base(
        video_reader, batch_size, video_reader_params, resize_params={}):
    pipeline = dali.pipeline.Pipeline(
        batch_size=batch_size, **pipeline_params)

    with pipeline:
        outputs = video_reader(
            **video_reader_params, **resize_params)
        pipeline.set_outputs(outputs)
    pipeline.build()

    return pipeline


def video_reader_resize_pipeline(batch_size, video_reader_params, resize_params):
    return video_reader_pipeline_base(
        dali.fn.video_reader_resize, batch_size, video_reader_params, resize_params)


def video_reader_pipeline(batch_size, video_reader_params):
    return video_reader_pipeline_base(
        dali.fn.video_reader, batch_size, video_reader_params)


def ground_truth_pipeline(batch_size, video_reader_params, resize_params):
    def get_next_frame():
        pipeline = video_reader_pipeline(
            batch_size, video_reader_params)

        pipe_out = pipeline.run()
        sequences_out = pipe_out[0].as_cpu().as_array()
        for sample in range(batch_size):
            yield [sequences_out[sample]]

    gt_pipeline = dali.pipeline.Pipeline(
        batch_size=video_reader_params['sequence_length'], **pipeline_params)

    with gt_pipeline:
        resized_frame = dali.fn.external_source(
            source=get_next_frame, num_outputs=1)
        resized_frame = resized_frame[0].gpu()
        resized_frame = dali.fn.resize(
            resized_frame, **resize_params)
        gt_pipeline.set_outputs(resized_frame)
    gt_pipeline.build()

    return gt_pipeline


def compare_video_resize_pipelines(pipeline, gt_pipeline, batch_size, video_length):
    global_sample_id = 0
    batch, = pipeline.run()
    batch = batch.as_cpu()
    for sample_id in range(batch_size):
        global_sample_id = global_sample_id + 1
        sample = batch.at(sample_id)
        gt_sample = gt_pipeline.run()[0].as_cpu().as_array()
        for frame_id in range(video_reader_params['sequence_length']):
            frame = sample[frame_id]
            gt_frame = gt_sample[frame_id]

            if gt_frame.shape == frame.shape:
                assert (gt_frame == frame).all(), "Images are not equal"
            else:
                assert (gt_frame.shape == frame.shape), "Shapes are not equal: {} != {}".format(
                    gt_frame.shape, frame.shape)

            # save_frame(gt_frame, global_sample_id, frame_id, 'gt')
            # save_frame(frame, global_sample_id, frame_id, 'frame')


def test_video_resize(batch_size=16):
    pipeline = video_reader_resize_pipeline(
        batch_size, video_reader_params, resize_params)

    gt_pipeline = ground_truth_pipeline(
        batch_size, video_reader_params, resize_params)

    compare_video_resize_pipelines(
        pipeline, gt_pipeline, batch_size, video_reader_params['sequence_length'])
