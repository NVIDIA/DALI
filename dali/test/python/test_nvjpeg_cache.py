# Copyright (c) 2019-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
import tempfile

import numpy as np
from PIL import Image
from nvidia.dali import fn, pipeline_def
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from numpy.testing import assert_array_equal
from test_utils import get_dali_extra_path
from nose2.tools import params

seed = 1549361629

img_root = get_dali_extra_path()
image_dir = img_root + "/db/single/jpeg"
batch_size = 20


def compare(tl1, tl2):
    tl1_cpu = tl1.as_cpu()
    tl2_cpu = tl2.as_cpu()
    assert len(tl1_cpu) == len(tl2_cpu)
    for i in range(0, len(tl1_cpu)):
        assert_array_equal(
            tl1_cpu.at(i), tl2_cpu.at(i), f"cached and non-cached images differ for sample #{i}"
        )


class HybridDecoderPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, cache_size, decoder_type):
        super(HybridDecoderPipeline, self).__init__(batch_size, num_threads, device_id, seed=seed)
        self.input = ops.readers.File(file_root=image_dir)
        policy = None
        if cache_size > 0:
            policy = "threshold"
        print("Decoder type:", decoder_type)
        decoder_module = (
            ops.experimental.decoders if "experimental" in decoder_type else ops.decoders
        )
        self.decode = decoder_module.Image(
            device="mixed",
            output_type=types.RGB,
            cache_size=cache_size,
            cache_type=policy,
            cache_debug=False,
            cache_batch_copy=True,
        )

    def define_graph(self):
        jpegs, labels = self.input(name="Reader")
        images = self.decode(jpegs)
        return (images, labels)


@params(("legacy",), ("experimental",))
def test_nvjpeg_cached(decoder_type):
    ref_pipe = HybridDecoderPipeline(batch_size, 1, 0, 0, decoder_type)
    cached_pipe = HybridDecoderPipeline(batch_size, 1, 0, 100, decoder_type)
    epoch_size = ref_pipe.epoch_size("Reader")

    for i in range(0, (2 * epoch_size + batch_size - 1) // batch_size):
        print("Batch %d-%d / %d" % (i * batch_size, (i + 1) * batch_size, epoch_size))
        ref_images, _ = ref_pipe.run()
        out_images, _ = cached_pipe.run()
        compare(ref_images, out_images)
        ref_images, _ = ref_pipe.run()
        out_images, _ = cached_pipe.run()
        compare(ref_images, out_images)
        ref_images, _ = ref_pipe.run()
        out_images, _ = cached_pipe.run()
        compare(ref_images, out_images)


def _make_dataset(root, images):
    """Creates a dataset with `images`: {name: (color, (height, width))} in a single class."""
    class_dir = os.path.join(root, "class0")
    os.makedirs(class_dir)
    refs = {}
    for name, (color, size) in images.items():
        img = np.empty((*size, 3), dtype=np.uint8)
        img[:] = color
        Image.fromarray(img).save(os.path.join(class_dir, name))
        refs[name] = img
    return refs


@pipeline_def(batch_size=1, num_threads=1, device_id=0, seed=seed)
def _cached_pipeline(
    root, decoder_type, batch_permutation=None, cast_encoded=False, skip_cached_images=False
):
    encoded, _ = fn.readers.file(file_root=root, skip_cached_images=skip_cached_images)
    if batch_permutation is not None:
        encoded = fn.permute_batch(encoded, indices=batch_permutation)
    if cast_encoded:
        encoded = fn.cast(encoded, dtype=types.UINT8)
    decoder = (
        fn.experimental.decoders.image if "experimental" in decoder_type else fn.decoders.image
    )
    return decoder(
        encoded,
        device="mixed",
        output_type=types.RGB,
        cache_size=16,
        cache_type="threshold",
        cache_threshold=0,
    )


@params(("legacy",), ("experimental",))
def test_cache_is_not_shared_between_file_roots(decoder_type):
    # Files with the same name relative to a different `file_root` are different files - the
    # decoded image cache, which is shared between the pipelines, must not mix them up.
    with tempfile.TemporaryDirectory() as tmp:
        root_a = os.path.join(tmp, "a")
        root_b = os.path.join(tmp, "b")
        (ref_a,) = _make_dataset(root_a, {"shared.png": ((250, 10, 10), (8, 8))}).values()
        (ref_b,) = _make_dataset(root_b, {"shared.png": ((10, 20, 240), (3, 5))}).values()
        pipe_a = _cached_pipeline(root=root_a, decoder_type=decoder_type)
        pipe_b = _cached_pipeline(root=root_b, decoder_type=decoder_type)
        for _ in range(2):  # the second iteration is served from the cache
            for pipe, ref in ((pipe_a, ref_a), (pipe_b, ref_b), (pipe_a, ref_a), (pipe_b, ref_b)):
                (out,) = pipe.run()
                assert_array_equal(out.as_cpu().at(0), ref)


@params(("legacy",), ("experimental",))
def test_cache_key_follows_the_sample_when_batch_is_permuted(decoder_type):
    # The cache key must stay attached to the data it describes when samples are reordered
    # between the reader and the decoder.
    with tempfile.TemporaryDirectory() as tmp:
        refs = _make_dataset(
            tmp, {"a.png": ((250, 10, 10), (8, 8)), "b.png": ((10, 20, 240), (3, 5))}
        )
        ref_a, ref_b = refs["a.png"], refs["b.png"]
        kwargs = dict(batch_size=2, root=tmp, decoder_type=decoder_type)
        pipe_swapped = _cached_pipeline(batch_permutation=[1, 0], **kwargs)
        pipe_ordered = _cached_pipeline(**kwargs)
        for _ in range(2):  # the second iteration is served from the cache
            for pipe, expected in ((pipe_swapped, (ref_b, ref_a)), (pipe_ordered, (ref_a, ref_b))):
                (out,) = pipe.run()
                out = out.as_cpu()
                assert_array_equal(out.at(0), expected[0])
                assert_array_equal(out.at(1), expected[1])


@params(("legacy",), ("experimental",))
def test_cache_key_survives_source_info_propagation(decoder_type):
    # Cast preserves the bytes but relies on the executor to propagate source metadata.
    # Warm the shared cache first so the second reader skips loading the encoded sample.
    with tempfile.TemporaryDirectory() as tmp:
        (ref,) = _make_dataset(tmp, {"a.png": ((250, 10, 10), (8, 8))}).values()
        warmup = _cached_pipeline(root=tmp, decoder_type=decoder_type)
        (out,) = warmup.run()
        assert_array_equal(out.as_cpu().at(0), ref)
        pipe = _cached_pipeline(
            root=tmp, decoder_type=decoder_type, cast_encoded=True, skip_cached_images=True
        )
        for _ in range(2):
            (out,) = pipe.run()
            assert_array_equal(out.as_cpu().at(0), ref)


def main():
    test_nvjpeg_cached("legacy")
    test_nvjpeg_cached("experimental")


if __name__ == "__main__":
    main()
