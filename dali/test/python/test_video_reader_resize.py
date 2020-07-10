import os
from functools import partial

import numpy as np
import nvidia.dali as dali
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline


video_directory = '/tmp/labelled_videos/'
video_directory_multiple_resolutions = '/tmp/video_resolution/'

pipeline_params = {
    'num_threads': 8,
    'device_id': 0,
    'seed': 0
}


video_reader_params = [{
    'device': 'gpu',
    'file_root': video_directory,
    'sequence_length': 32,
    'random_shuffle': False
}, {
    'device': 'gpu',
    'file_root': video_directory_multiple_resolutions,
    'sequence_length': 32,
    'random_shuffle': False
}]


resize_params = [{
    'resize_x': 300,
    'resize_y': 200,
    'interp_type': types.DALIInterpType.INTERP_CUBIC,
    'minibatch_size': 8
}, {
    'resize_x': 300,
    'interp_type': types.DALIInterpType.INTERP_CUBIC,
    'minibatch_size': 8
}, {
    'resize_x': 300,
    'resize_y': 200,
    'interp_type': types.DALIInterpType.INTERP_LANCZOS3,
    'minibatch_size': 8
}, {
    'resize_shorter': 300,
    'interp_type': types.DALIInterpType.INTERP_CUBIC,
    'minibatch_size': 8
}, {
    'resize_longer': 500,
    'interp_type': types.DALIInterpType.INTERP_CUBIC,
    'minibatch_size': 8
}, {
    'resize_x': 300,
    'resize_y': 200,
    'min_filter': types.DALIInterpType.INTERP_CUBIC,
    'mag_filter': types.DALIInterpType.INTERP_TRIANGULAR,
    'minibatch_size': 8
}, {
    'resize_x': 300,
    'resize_y': 200,
    'interp_type': types.DALIInterpType.INTERP_CUBIC,
    'minibatch_size': 4
}]


def video_reader_resize_pipeline(batch_size, video_reader_params, resize_params):
    pipeline = dali.pipeline.Pipeline(
        batch_size=batch_size, **pipeline_params)

    with pipeline:
        outputs = dali.fn.video_reader_resize(
            **video_reader_params, **resize_params)
        if type(outputs) == list:
            outputs = outputs[0]
        pipeline.set_outputs(outputs)
    pipeline.build()

    return pipeline


def ground_truth_pipeline(batch_size, video_reader_params, resize_params):
    video_length = video_reader_params['sequence_length']
    class VideoReaderPipeline(Pipeline):
        def __init__(self):
            super(VideoReaderPipeline, self).__init__(batch_size, **pipeline_params)
            
            self.reader = ops.VideoReader(**video_reader_params)
            self.element_extract = ops.ElementExtract(device='gpu', element_map=list(range(video_length)))
            self.resize = ops.Resize(device='gpu', **resize_params)

        def define_graph(self):
            video, _ = self.reader(name='Reader')
            resized_frames = self.element_extract(video)

            for i in range(video_length):
                resized_frames[i] = self.resize(resized_frames[i])
            return resized_frames

    pipeline = VideoReaderPipeline()
    pipeline.build()

    return pipeline


def compare_video_resize_pipelines(pipeline, gt_pipeline, batch_size, video_length, iterations=16):
    for i in range(iterations):
        batch, = pipeline.run()
        batch = batch.as_cpu()
        gt_batch = list(gt_pipeline.run())

        for i in range(video_length):
            gt_batch[i] = gt_batch[i].as_cpu()

        for sample_id in range(batch_size):
            sample = batch.at(sample_id)
            for frame_id in range(video_length):
                frame = sample[frame_id]
                gt_frame = gt_batch[frame_id].at(sample_id)

                if gt_frame.shape == frame.shape:
                    assert (gt_frame == frame).all(), "Images are not equal"
                else:
                    assert (gt_frame.shape == frame.shape), "Shapes are not equal: {} != {}".format(
                        gt_frame.shape, frame.shape)


def run_for_params(batch_size, video_reader_params, resize_params):
    pipeline = video_reader_resize_pipeline(
        batch_size, video_reader_params, resize_params)

    gt_pipeline = ground_truth_pipeline(
        batch_size, video_reader_params, resize_params)

    compare_video_resize_pipelines(
        pipeline, gt_pipeline, batch_size, video_reader_params['sequence_length'])


def test_video_resize(batch_size=2):
    for vp in video_reader_params:
        for rp in resize_params:
            yield run_for_params, batch_size, vp, rp
