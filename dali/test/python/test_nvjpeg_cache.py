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


def main():
    test_nvjpeg_cached("legacy")
    test_nvjpeg_cached("experimental")


if __name__ == "__main__":
    main()
