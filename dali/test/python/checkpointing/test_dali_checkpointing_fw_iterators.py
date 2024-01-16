# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from checkpointing.test_dali_checkpointing import (
    warmup_epochs,
    pipeline_args,
    check_single_input_operator_pipeline,
    make_dummy_source,
    make_external_source_test_pipeline_factory,
    images_dir,
)
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import pipeline_def
from nose2.tools import params, cartesian_params

class FwTestBase:
    FwIterator = None

    def equal(self, a, b):
        raise NotImplementedError

    # Helpers

    def check_pipeline_checkpointing(self, pipeline_factory, reader_name=None, size=-1):
        pipe = pipeline_factory(**pipeline_args)
        pipe.build()

        iter = self.FwIterator(pipe, ["data"], auto_reset=True, reader_name=reader_name, size=size)
        for _ in range(warmup_epochs):
            for _ in iter:
                pass

        restored = pipeline_factory(**pipeline_args, checkpoint=iter.checkpoints()[0])
        restored.build()
        iter2 = self.FwIterator(restored, ["data"], auto_reset=True, reader_name=reader_name, size=size)

        for out1, out2 in zip(iter, iter2):
            for d1, d2 in zip(out1, out2):
                for key in d1.keys():
                    assert self.equal(d1[key], d2[key])


    def check_single_input_operator(self, op, device, **kwargs):
        pipeline_factory = check_single_input_operator_pipeline(op, device, **kwargs)
        self.check_pipeline_checkpointing(pipeline_factory, reader_name="Reader")


    def check_no_input_operator(self, op, device, **kwargs):
        @pipeline_def
        def pipeline_factory():
            return op(device=device, **kwargs)

        self.check_pipeline_checkpointing(pipeline_factory, size=8)


    # Reader tests section

    @params(
        (1, 3, 0, 1, True, False, False),
        (5, 10, 0, 2, True, False, False),
        (3, 64, 3, 4, False, False, False),
        (0, 32, 1, 4, False, False, True),
        (3, 64, 3, 4, False, False, True),
        (1, 8, 0, 2, False, True, False),
        (1, 8, 1, 2, False, True, False),
        (1, 8, 3, 4, False, True, False),
        (1, 3, 0, 1, True, False, False, 1),
        (5, 10, 0, 2, True, False, False, 2),
        (3, 64, 3, 4, False, False, True, 3),
    )
    def test_file_reader(
        self,
        num_epochs,
        batch_size,
        shard_id,
        num_shards,
        random_shuffle,
        shuffle_after_epoch,
        stick_to_shard,
        iters_into_epoch=None,
    ):
        @pipeline_def(batch_size=batch_size, device_id=0, num_threads=4, enable_checkpointing=True)
        def pipeline():
            data, label = fn.readers.file(
                name="Reader",
                file_root=images_dir,
                pad_last_batch=True,
                random_shuffle=random_shuffle,
                shard_id=shard_id,
                num_shards=num_shards,
                shuffle_after_epoch=shuffle_after_epoch,
                stick_to_shard=stick_to_shard,
            )
            image = fn.decoders.image_random_crop(data, device="mixed")
            image = fn.resize(image, size=(200, 200))
            return image, label

        p = pipeline()
        p.build()

        iter = self.FwIterator(p, ["data", "labels"], auto_reset=True, reader_name="Reader")
        for epoch in range(num_epochs):
            for i, _ in enumerate(iter):
                if iters_into_epoch is not None:
                    if epoch == num_epochs - 1 and i == iters_into_epoch - 1:
                        break

        restored = pipeline(checkpoint=iter.checkpoints()[0])
        restored.build()
        iter2 = self.FwIterator(restored, ["data", "labels"], auto_reset=True, reader_name="Reader")

        for out1, out2 in zip(iter, iter2):
            for d1, d2 in zip(out1, out2):
                for key in d1.keys():
                    assert self.equal(d1[key], d2[key])

    # Random operators section

    @cartesian_params(("cpu", "gpu"), (None, (1,), (10,)))
    def test_random_coin_flip(self, device, shape):
        self.check_no_input_operator(fn.random.coin_flip, device, shape=shape)

    @cartesian_params(("cpu", "gpu"), (None, (1,), (10,)))
    def test_random_normal(self, device, shape):
        self.check_no_input_operator(fn.random.normal, device, shape=shape)

    @cartesian_params(("cpu", "gpu"), (None, (1,), (10,)))
    def test_random_uniform(self, device, shape):
        self.check_no_input_operator(fn.random.uniform, device, shape=shape)

    # Stateless operators section

    @cartesian_params(("cpu", "gpu"), (None, (1,), (10,)))
    def test_constant(self, device, shape):
        self.check_no_input_operator(fn.constant, device, idata=42, shape=shape)

    # External source section

    def check_external_source_pipeline_checkpointing(
        self, pipeline_factory, iterations, *, size=-1
    ):
        def run(iterator, iterations):
            completed_iterations = 0
            while completed_iterations < iterations:
                for _ in iterator:
                    completed_iterations += 1
                    if completed_iterations == iterations:
                        break

        pipeline = pipeline_factory()
        pipeline.build()

        iter = self.FwIterator(pipeline, ["data"], auto_reset=True, size=size)

        run(iter, iterations)

        restored = pipeline_factory(checkpoint=iter.checkpoints()[0])
        restored.build()
        iter2 = self.FwIterator(restored, ["data"], auto_reset=True, size=size)

        for out1, out2 in zip(iter, iter2):
            for d1, d2 in zip(out1, out2):
                for key in d1.keys():
                    assert self.equal(d1[key], d2[key])

    @cartesian_params(
        ((1, 1), (4, 5)),  # (epoch size, batch size)
        (0, 4, 11),  # test iterations
        ("idx", "batch_info", "sample_info"),  # indexing mode
        (True, False),  # parallel
    )
    def test_external_source_checkpointing(self, dataset_info, iterations, mode, parallel):
        epoch_size, batch_size = dataset_info
        source = make_dummy_source(epoch_size, batch_size, mode)
        pf = make_external_source_test_pipeline_factory(source, mode, batch_size, parallel)
        self.check_external_source_pipeline_checkpointing(pf, iterations)


# Framework tests


class TestPytorch(FwTestBase):
    def __init__(self):
        from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator

        self.FwIterator = PyTorchIterator

    def equal(self, a, b):
        return (a == b).all()
