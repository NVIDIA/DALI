# Copyright (c) 2020-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import nvidia.dali.fn as fn
import numpy as np

from test_utils import RandomDataIterator


class CoordFlipPipeline(Pipeline):
    def __init__(
        self,
        device,
        batch_size,
        iterator,
        layout,
        center_x=None,
        center_y=None,
        center_z=None,
        num_threads=1,
        device_id=0,
    ):
        super(CoordFlipPipeline, self).__init__(batch_size, num_threads, device_id)
        self.device = device
        self.iterator = iterator
        self.coord_flip = ops.CoordFlip(
            device=self.device,
            layout=layout,
            center_x=center_x,
            center_y=center_y,
            center_z=center_z,
        )
        self.flip_x = ops.random.CoinFlip(probability=0.5)
        self.flip_y = ops.random.CoinFlip(probability=0.5)
        self.flip_z = ops.random.CoinFlip(probability=0.5) if len(layout) == 3 else None

    def define_graph(self):
        inputs = fn.external_source(lambda: next(self.iterator))
        inputs = 0.5 + inputs  # Make it fit the range [0.0, 1.0]
        out = inputs.gpu() if self.device == "gpu" else inputs
        flip_x = self.flip_x()
        flip_y = self.flip_y()
        flip_z = self.flip_z() if self.flip_z is not None else None
        out = self.coord_flip(out, flip_x=flip_x, flip_y=flip_y, flip_z=flip_z)
        outputs = [inputs, out, flip_x, flip_y]
        if flip_z is not None:
            outputs.append(flip_z)
        return outputs


def check_operator_coord_flip(device, batch_size, layout, shape, center_x, center_y, center_z):
    eii1 = RandomDataIterator(batch_size, shape=shape, dtype=np.float32)
    pipe = CoordFlipPipeline(device, batch_size, iter(eii1), layout, center_x, center_y, center_z)
    for _ in range(30):
        outputs = tuple(out.as_cpu() for out in pipe.run())
        for sample in range(batch_size):
            in_coords = outputs[0].at(sample)
            out_coords = outputs[1].at(sample)
            if in_coords.shape == () or in_coords.shape[0] == 0:
                assert out_coords.shape == () or out_coords.shape[0] == 0
                continue

            flip_x = outputs[2].at(sample)
            flip_y = outputs[3].at(sample)
            flip_z = None
            if len(layout) == 3:
                flip_z = outputs[4].at(sample)
            _, ndim = in_coords.shape

            flip_dim = [flip_x, flip_y]
            if ndim == 3:
                flip_dim.append(flip_z)

            center_dim = [center_x, center_y]
            if ndim == 3:
                center_dim.append(center_z)

            expected_out_coords = np.copy(in_coords)
            for d in range(ndim):
                if flip_dim[d]:
                    expected_out_coords[:, d] = 2 * center_dim[d] - in_coords[:, d]
            np.testing.assert_allclose(out_coords[:, d], expected_out_coords[:, d])


def test_operator_coord_flip():
    for device in ["cpu", "gpu"]:
        for batch_size in [1, 3]:
            layout_shape_values = [("x", (10, 1)), ("xy", (10, 2)), ("xyz", (10, 3))]
            if device == "cpu":
                layout_shape_values.append(("xy", (0, 2)))
            for layout, shape in layout_shape_values:
                for center_x, center_y, center_z in [(0.5, 0.5, 0.5), (0.0, 1.0, -0.5)]:
                    yield (
                        check_operator_coord_flip,
                        device,
                        batch_size,
                        layout,
                        shape,
                        center_x,
                        center_y,
                        center_z,
                    )
