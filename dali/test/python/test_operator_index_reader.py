import math
from nvidia.dali import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.fn as fn
import nvidia.dali.tfrecord as tfrec
import os.path
import tempfile
import numpy as np
from test_utils import get_dali_extra_path
from nose.tools import assert_raises

def skip_second(src, dst):
    with open(src, 'r') as tmp_f:
        with open(dst, 'w') as f:
            second = False
            for l in tmp_f:
                if not second:
                    f.write(l)
                second = not second

def test_tfrecord():
    class TFRecordPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus, data, data_idx):
            super(TFRecordPipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.TFRecordReader(path = data,
                                            index_path = data_idx,
                                            features = {"image/encoded" : tfrec.FixedLenFeature((), tfrec.string, ""),
                                                        "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64,  -1)
                                            })

        def define_graph(self):
            inputs = self.input(name="Reader")
            images = inputs["image/encoded"]
            return images

    tfrecord = os.path.join(get_dali_extra_path(), 'db', 'tfrecord', 'train')
    tfrecord_idx_org = os.path.join(get_dali_extra_path(), 'db', 'tfrecord', 'train.idx')
    tfrecord_idx = "tfr_train.idx"

    idx_files_dir = tempfile.TemporaryDirectory()
    idx_file = os.path.join(idx_files_dir.name, tfrecord_idx)

    skip_second(tfrecord_idx_org, idx_file)

    pipe = TFRecordPipeline(1, 1, 0, 1, tfrecord, idx_file)
    pipe_org = TFRecordPipeline(1, 1, 0, 1, tfrecord, tfrecord_idx_org)
    pipe.build()
    pipe_org.build()
    iters = pipe.epoch_size("Reader")
    for _ in  range(iters):
        out = pipe.run()
        out_ref = pipe_org.run()
        for a, b in zip(out, out_ref):
            assert np.array_equal(a.as_array(), b.as_array())
        _ = pipe_org.run()

def test_recordio():
    class MXNetReaderPipeline(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus, data, data_idx):
            super(MXNetReaderPipeline, self).__init__(batch_size, num_threads, device_id)
            self.input = ops.MXNetReader(path = [data], index_path=[data_idx],
                                        shard_id = device_id, num_shards = num_gpus)

        def define_graph(self):
            images, _ = self.input(name="Reader")
            return images

    recordio = os.path.join(get_dali_extra_path(), 'db', 'recordio', 'train.rec')
    recordio_idx_org = os.path.join(get_dali_extra_path(), 'db', 'recordio', 'train.idx')
    recordio_idx = "rio_train.idx"

    idx_files_dir = tempfile.TemporaryDirectory()
    idx_file = os.path.join(idx_files_dir.name, recordio_idx)

    skip_second(recordio_idx_org, idx_file)

    pipe = MXNetReaderPipeline(1, 1, 0, 1, recordio, idx_file)
    pipe_org = MXNetReaderPipeline(1, 1, 0, 1, recordio, recordio_idx_org)
    pipe.build()
    pipe_org.build()
    iters = pipe.epoch_size("Reader")
    for _ in  range(iters):
        out = pipe.run()
        out_ref = pipe_org.run()
        for a, b in zip(out, out_ref):
            assert np.array_equal(a.as_array(), b.as_array())
        _ = pipe_org.run()

def test_wrong_feature_shape():
    features = {
        'image/encoded': tfrec.FixedLenFeature((), tfrec.string, ""),
        'image/object/bbox': tfrec.FixedLenFeature([], tfrec.float32, -1.0),
        'image/object/class/label': tfrec.FixedLenFeature([], tfrec.int64, -1),
    }
    test_dummy_data_path = os.path.join(get_dali_extra_path(), 'db', 'coco_dummy')
    pipe = Pipeline(1, 1, 0)
    with pipe:
        input = fn.tfrecord_reader(path = os.path.join(test_dummy_data_path, 'small_coco.tfrecord'),
                                   index_path = os.path.join(test_dummy_data_path, 'small_coco_index.idx'),
                                   features = features)
    pipe.set_outputs(input['image/encoded'], input['image/object/class/label'], input['image/object/bbox'])
    pipe.build()
    assert_raises(RuntimeError, pipe.run)