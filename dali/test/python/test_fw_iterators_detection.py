# Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import nvidia.dali.ops as ops
import os
from nvidia.dali.pipeline import Pipeline

from test_utils import get_dali_extra_path
from nose_utils import assert_raises

DALI_EXTRA_PATH = get_dali_extra_path()
EPOCH_SIZE = 32
BATCH_SIZE = 1


class DetectionPipeline(Pipeline):
    def __init__(self, batch_size, device_id, file_root, annotations_file):
        super().__init__(batch_size, 2, device_id, True, 12)

        # Reading COCO dataset
        self.input = ops.readers.COCO(
            file_root=file_root,
            annotations_file=annotations_file,
            shard_id=device_id,
            num_shards=1,
            ratio=True,
            ltrb=True,
        )

    def define_graph(self):
        inputs, boxes, labels = self.input(name="Reader")
        return inputs, boxes.gpu(), labels


def data_paths():
    root = os.path.join(DALI_EXTRA_PATH, "db", "coco", "images")
    annotations = os.path.join(DALI_EXTRA_PATH, "db", "coco", "instances.json")
    return root, annotations


##############
# Unit tests #
##############


def test_mxnet_pipeline_dynamic_shape():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator

    root, annotations = data_paths()
    pipeline = DetectionPipeline(BATCH_SIZE, 0, root, annotations)
    train_loader = MXNetIterator(
        [pipeline],
        [
            ("data", MXNetIterator.DATA_TAG),
            ("bboxes", MXNetIterator.LABEL_TAG),
            ("label", MXNetIterator.LABEL_TAG),
        ],
        EPOCH_SIZE,
        auto_reset=False,
        dynamic_shape=True,
    )
    for data in train_loader:
        assert data is not None


def test_pytorch_pipeline_dynamic_shape():
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

    root, annotations = data_paths()
    pipeline = DetectionPipeline(BATCH_SIZE, 0, root, annotations)
    train_loader = PyTorchIterator(
        [pipeline], ["data", "bboxes", "label"], EPOCH_SIZE, auto_reset=False, dynamic_shape=True
    )
    for data in train_loader:
        assert data is not None


def test_paddle_pipeline_dynamic_shape():
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator

    root, annotations = data_paths()
    pipeline = DetectionPipeline(BATCH_SIZE, 0, root, annotations)
    train_loader = PaddleIterator(
        [pipeline], ["data", "bboxes", "label"], EPOCH_SIZE, auto_reset=False, dynamic_shape=True
    )
    for data in train_loader:
        assert data is not None


def test_api_fw_check1_pytorch():
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

    yield from test_api_fw_check1(PyTorchIterator, ["data", "bboxes", "label"])


def test_api_fw_check1_mxnet():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator

    yield from test_api_fw_check1(
        MXNetIterator,
        [
            ("data", MXNetIterator.DATA_TAG),
            ("bboxes", MXNetIterator.LABEL_TAG),
            ("label", MXNetIterator.LABEL_TAG),
        ],
    )


def test_api_fw_check1_paddle():
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator

    yield from test_api_fw_check1(PaddleIterator, ["data", "bboxes", "label"])


def test_api_fw_check1(iter_type, data_definition):
    root, annotations = data_paths()
    pipe = DetectionPipeline(BATCH_SIZE, 0, root, annotations)
    train_loader = iter_type(
        [pipe], data_definition, EPOCH_SIZE, auto_reset=False, dynamic_shape=True
    )
    train_loader.__next__()
    for method in [
        pipe.schedule_run,
        pipe.share_outputs,
        pipe.release_outputs,
        pipe.outputs,
        pipe.run,
    ]:
        with assert_raises(
            RuntimeError,
            glob="Mixing pipeline API type. Currently used: PipelineAPIType.ITERATOR,"
            " but trying to use PipelineAPIType.*",
        ):
            method()
    # disable check
    pipe.enable_api_check(False)
    for method in [
        pipe.schedule_run,
        pipe.share_outputs,
        pipe.release_outputs,
        pipe.outputs,
        pipe.run,
    ]:
        try:
            method()
        except RuntimeError:
            assert False
    yield check, iter_type


def test_api_fw_check2_mxnet():
    from nvidia.dali.plugin.mxnet import DALIGenericIterator as MXNetIterator

    yield from test_api_fw_check2(
        MXNetIterator,
        [
            ("data", MXNetIterator.DATA_TAG),
            ("bboxes", MXNetIterator.LABEL_TAG),
            ("label", MXNetIterator.LABEL_TAG),
        ],
    )


def test_api_fw_check2_pytorch():
    from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

    yield from test_api_fw_check2(PyTorchIterator, ["data", "bboxes", "label"])


def test_api_fw_check2_paddle():
    from nvidia.dali.plugin.paddle import DALIGenericIterator as PaddleIterator

    yield from test_api_fw_check2(PaddleIterator, ["data", "bboxes", "label"])


def test_api_fw_check2(iter_type, data_definition):
    root, annotations = data_paths()

    pipe = DetectionPipeline(BATCH_SIZE, 0, root, annotations)
    pipe.schedule_run()
    pipe.share_outputs()
    pipe.release_outputs()
    pipe.schedule_run()
    pipe.outputs()
    with assert_raises(
        RuntimeError,
        glob=(
            "Mixing pipeline API type. Currently used: PipelineAPIType.SCHEDULED,"
            " but trying to use PipelineAPIType.ITERATOR"
        ),
    ):
        train_loader = iter_type(
            [pipe], data_definition, EPOCH_SIZE, auto_reset=False, dynamic_shape=True
        )
        train_loader.__next__()
    # disable check
    pipe.enable_api_check(False)
    try:
        train_loader = iter_type(
            [pipe], data_definition, EPOCH_SIZE, auto_reset=False, dynamic_shape=True
        )
        train_loader.__next__()
        assert True
    except RuntimeError:
        assert False
    yield check, iter_type


def check(iter_type):
    pass
