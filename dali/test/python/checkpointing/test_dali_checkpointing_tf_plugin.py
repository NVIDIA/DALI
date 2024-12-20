from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
from nvidia.dali.types import DALIDataType
import nvidia.dali.plugin.tf as dali_tf
import tensorflow as tf
import numpy as np
import tempfile
from test_utils import get_dali_extra_path
import os
from nose_utils import assert_raises

data_root = get_dali_extra_path()
images_dir = os.path.join(data_root, "db", "single", "jpeg")


def check_dataset_checkpointing(dali_dataset, *, warmup_iters, test_iters):
    it = iter(dali_dataset)
    mgr = tf.train.Checkpoint(it)

    def read_data(it, iters):
        data = None
        for _ in range(iters):
            out = next(it)
            if data is None:
                data = [[] for _ in range(len(out))]
            assert len(data) == len(out)
            for i, x in enumerate(out):
                data[i].append(np.asarray(x))
        return data

    def compare_data(data1, data2):
        assert len(data1) == len(data2)
        for output1, output2 in zip(data1, data2):
            assert len(output1) == len(output2)
            for x1, x2 in zip(output1, output2):
                assert (x1 == x2).all()

    read_data(it, warmup_iters)

    with tempfile.TemporaryDirectory() as cpt_dir:
        cpt = mgr.save(cpt_dir)
        data = read_data(it, test_iters)
        mgr.restore(cpt)

    data_restored = read_data(it, test_iters)
    compare_data(data, data_restored)


def check_pipeline_checkpointing(pipeline_factory, output_dtypes, **kwargs):
    p = pipeline_factory()
    with tf.device("cpu"):
        dataset = dali_tf.DALIDataset(pipeline=p, output_dtypes=output_dtypes)
        check_dataset_checkpointing(dataset, **kwargs)


def test_random():
    @pipeline_def(num_threads=4, device_id=0, batch_size=4, enable_checkpointing=True)
    def pipeline():
        return fn.random.uniform(dtype=DALIDataType.FLOAT)

    check_pipeline_checkpointing(pipeline, (tf.float32,), warmup_iters=7, test_iters=10)


def test_reader():
    @pipeline_def(num_threads=4, device_id=0, batch_size=4, enable_checkpointing=True)
    def pipeline():
        jpeg, label = fn.readers.file(
            file_root=images_dir, pad_last_batch=False, random_shuffle=True
        )
        return (jpeg, label)

    check_pipeline_checkpointing(pipeline, (tf.uint8, tf.int32), warmup_iters=7, test_iters=10)


def test_inputs_unsupported():
    @pipeline_def(num_threads=4, device_id=0, batch_size=4, enable_checkpointing=True)
    def external_source_pipe():
        return fn.external_source(source=lambda x: np.array(x.iteration), batch=False)

    p = external_source_pipe()
    with tf.device("cpu"):
        dataset = dali_tf.experimental.DALIDatasetWithInputs(
            pipeline=p,
            output_dtypes=(tf.int64,),
            batch_size=5,
            output_shapes=(5,),
            num_threads=4,
            device_id=0,
        )
        with assert_raises(
            Exception, regex="Checkpointing is not supported for DALI dataset with inputs."
        ):
            check_dataset_checkpointing(dataset, warmup_iters=1, test_iters=1)
