# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


import glob
import nvidia.dali.fn as fn
import nvidia.dali.experimental.dynamic as ndd
import os
from nose2.tools import params
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.tfrecord as tfrec
import test_utils
from ndd_vs_fn_test_utils import MAX_BATCH_SIZE, N_ITERATIONS, compare, sign_off
from webdataset_base import generate_temp_index_file as generate_temp_wds_index


def run_reader_test(fn_reader, ndd_reader, device, batch_size=MAX_BATCH_SIZE, reader_args={}):

    pipe_reader = create_reader_pipeline(fn_reader, device, batch_size, reader_args)
    ndd_reader = create_reader_ndd(ndd_reader, device, reader_args)

    for batch in ndd_reader.next_epoch(batch_size=batch_size):
        pipe_out = pipe_reader.run()
        compare(pipe_out, batch)

    name = ndd_reader._op_path
    assert name in sign_off.tested_ops, f"Operator {name} tested but not registered!"


def create_reader_pipeline(fn_reader, device, batch_size=MAX_BATCH_SIZE, reader_args={}):
    @pipeline_def(batch_size=batch_size, num_threads=ndd.get_num_threads(), device_id=0)
    def pipeline():
        output = fn_reader(device=device, **reader_args)
        if isinstance(output, list):  # DALI uses list to collect operator multi-output
            return tuple(output[i] for i in range(len(output)))
        else:
            return output

    pipe = pipeline()
    pipe.build()

    return pipe


def create_reader_ndd(ndd_reader, device, reader_args={}):
    reader = ndd_reader(device=device, **reader_args)
    return reader


data_root = test_utils.get_dali_extra_path()
files = (
    [os.path.join(data_root, "db", "single", "jpeg", "134", "baukran-3703469_1280.jpg")]
    * MAX_BATCH_SIZE
    * N_ITERATIONS
)
images_dir = os.path.join(data_root, "db", "single", "jpeg")
caffe_dir = os.path.join(data_root, "db", "lmdb")
caffe2_dir = os.path.join(data_root, "db", "c2lmdb")
recordio_dir = os.path.join(data_root, "db", "recordio")
tfrecord_dir = os.path.join(data_root, "db", "tfrecord")
webdataset_dir = os.path.join(data_root, "db", "webdataset")
coco_dir = os.path.join(data_root, "db", "coco", "images")
coco_annotation = os.path.join(data_root, "db", "coco", "instances.json")
video_files = [
    os.path.join(test_utils.get_dali_extra_path(), "db", "video", "vfr", "test_1.mp4"),
    os.path.join(test_utils.get_dali_extra_path(), "db", "video", "vfr", "test_2.mp4"),
]
vid_dir = os.path.join(data_root, "db", "video", "sintel", "video_files")
vid_files = ["sintel_trailer-720p_2.mp4"]
vid_filenames = [os.path.join(vid_dir, vid_file) for vid_file in vid_files]


def _expand_reader_test_cases(test_cases):
    ret = []
    for case in test_cases:
        fn_op, ndd_op, params = case
        sign_off.register_test(ndd_op._op_path)
        for dev in ndd_op._supported_backends:
            ret.append((ndd_op._op_name, fn_op, ndd_op, dev, params))
    return ret


READERS = _expand_reader_test_cases(
    [
        (
            fn.experimental.readers.video,
            ndd.experimental.readers.Video,
            {"filenames": video_files, "sequence_length": 3},
        ),
        (fn.readers.caffe, ndd.readers.Caffe, {"path": caffe_dir}),
        (fn.readers.caffe2, ndd.readers.Caffe2, {"path": caffe2_dir}),
        (
            fn.readers.coco,
            ndd.readers.COCO,
            {"file_root": coco_dir, "annotations_file": coco_annotation},
        ),
        (fn.readers.file, ndd.readers.File, {"files": files}),
        (
            fn.readers.mxnet,
            ndd.readers.MXNet,
            {
                "path": os.path.join(recordio_dir, "train.rec"),
                "index_path": os.path.join(recordio_dir, "train.idx"),
            },
        ),
        (
            fn.readers.video,
            ndd.readers.Video,
            {
                "filenames": [
                    os.path.join(
                        test_utils.get_dali_extra_path(), "db", "video", "cfr", "test_1.mp4"
                    )
                ],
                "sequence_length": 3,
            },
        ),
        # (
        #     fn.readers.video_resize,
        #     ndd.readers.VideoResize,
        #     {
        #         "filenames": vid_filenames,
        #         "sequence_length": 31,
        #         "roi_start": (90, 0),
        #         "roi_end": (630, 1280),
        #         "file_list_include_preceding_frame": True,
        #     },
        # ),  # BUG
    ]
)


@params(*READERS)
def test_readers(op_name, fn_reader, ndd_reader, device, reader_args):
    run_reader_test(
        fn_reader=fn_reader, ndd_reader=ndd_reader, device=device, reader_args=reader_args
    )


@sign_off("readers.Webdataset")
def test_webdataset_reader():
    webdataset = os.path.join(webdataset_dir, "MNIST", "devel-0.tar")
    webdataset_idx = generate_temp_wds_index(webdataset)
    run_reader_test(
        fn_reader=fn.readers.webdataset,
        ndd_reader=ndd.readers.Webdataset,
        device="cpu",
        reader_args={
            "paths": webdataset,
            "index_paths": webdataset_idx.name,
            "ext": ["jpg", "cls"],
        },
    )


def test_tfrecord_reader():
    tfrecord = sorted(glob.glob(os.path.join(tfrecord_dir, "*[!i][!d][!x]")))
    tfrecord_idx = sorted(glob.glob(os.path.join(tfrecord_dir, "*idx")))

    @pipeline_def(batch_size=MAX_BATCH_SIZE, num_threads=ndd.get_num_threads(), device_id=0)
    def pipeline():
        inp = fn.readers.tfrecord(
            path=tfrecord,
            index_path=tfrecord_idx,
            shard_id=0,
            num_shards=1,
            features={
                "image/encoded": tfrec.FixedLenFeature((), tfrec.string, ""),
                "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64, -1),
            },
        )
        return inp["image/encoded"]

    pipe = pipeline()
    pipe.build()

    ndd_reader = ndd.readers.TFRecord(
        path=tfrecord,
        index_path=tfrecord_idx,
        shard_id=0,
        num_shards=1,
        features={
            "image/encoded": tfrec.FixedLenFeature((), tfrec.string, ""),
            "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64, -1),
        },
    )

    for batch in ndd_reader.next_epoch(batch_size=MAX_BATCH_SIZE):
        pipe_out = pipe.run()
        compare(pipe_out, batch["image/encoded"])
