# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import sys

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import numpy as np
import os
from test_utils import get_dali_extra_path


DEBUG_LVL = 0
SHOW_IMAGES = False

np.random.seed(1234)

data_root = get_dali_extra_path()
img_dir = os.path.join(data_root, 'db', 'single', 'jpeg')


np_type_map = {
    types.UINT8: np.uint8,
    types.UINT16: np.uint16,
    types.UINT32: np.uint32,
    types.UINT64: np.uint64,
    types.FLOAT16: np.float16,
    types.FLOAT: np.float32,
    types.FLOAT64: np.float64,
    types.INT8: np.int8,
    types.INT16: np.int16,
    types.INT32: np.int32,
    types.INT64: np.int64,
}


def intersects(anchors1, shapes1, anchors2, shapes2):
    for i in range(len(anchors1)):
        if anchors1[i] + shapes1[i] <= anchors2[i] or anchors2[i] + shapes2[i] <= anchors1[i]:
            return False
    return True


def prepare_cuts(
        iters=4,
        batch_size=16,
        input_size=None,
        output_size=None,
        even_paste_count=False,
        no_intersections=False,
        full_input=False,
        in_anchor_top_left=False,
        in_anchor_range=None,
        out_anchor_top_left=False,
        out_anchor_range=None,
        out_of_bounds_count=0,
):
    # Those two will not work together
    assert(out_of_bounds_count == 0 or not no_intersections)

    in_idx_l = [np.zeros(shape=(0,), dtype=np.int32) for _ in range(batch_size)]
    in_anchors_l = [np.zeros(shape=(0, 2), dtype=np.int32) for _ in range(batch_size)]
    shapes_l = [np.zeros(shape=(0, 2), dtype=np.int32) for _ in range(batch_size)]
    out_anchors_l = [np.zeros(shape=(0, 2), dtype=np.int32) for _ in range(batch_size)]
    assert len(input_size) == len(output_size)
    dim = len(input_size)
    for i in range(batch_size):
        for j in range(iters):
            while True:
                in_idx = np.int32(np.random.randint(batch_size))
                out_idx = np.int32(i if even_paste_count else np.random.randint(batch_size))
                shape = [np.int32(
                    np.random.randint(
                        min(input_size[i], output_size[i]) // (iters if no_intersections else 1)
                    ) + 1
                ) for i in range(dim)] if not full_input else input_size

                if in_anchor_top_left:
                    in_anchor = [0] * dim
                elif in_anchor_range is not None:
                    in_anchor = [np.int32(np.random.randint(in_anchor_range[0][i], in_anchor_range[1][i])) for i in range(dim)]
                    if full_input:
                        shape = [np.int32(input_size[i] - in_anchor[i]) for i in range(dim)]
                else:
                    in_anchor = [np.int32(np.random.randint(input_size[i] - shape[i] + 1)) for i in range(dim)]

                if out_anchor_top_left:
                    out_anchor = [0] * dim
                elif out_anchor_range is not None:
                    out_anchor = [np.int32(np.random.randint(out_anchor_range[0][i], out_anchor_range[1][i])) for i in range(dim)]
                else:
                    out_anchor = [np.int32(np.random.randint(output_size[i] - shape[i] + 1)) for i in range(dim)]

                if no_intersections:
                    is_ok = True
                    for k in range(len(in_idx_l[out_idx])):
                        if intersects(out_anchors_l[out_idx][k], shapes_l[out_idx][k], out_anchor, shape):
                            is_ok = False
                            break
                    if not is_ok:
                        continue
                    break
                break

            if DEBUG_LVL >= 1:
                print(f"""in_idx: {in_idx}, out_idx: {out_idx}, in_anchor: {
                in_anchor}, in_shape: {shape}, out_anchor: {out_anchor}""")

            in_idx_l[out_idx] = np.append(in_idx_l[out_idx], [in_idx], axis=0)
            in_anchors_l[out_idx] = np.append(in_anchors_l[out_idx], [in_anchor], axis=0)
            shapes_l[out_idx] = np.append(shapes_l[out_idx], [shape], axis=0)
            out_anchors_l[out_idx] = np.append(out_anchors_l[out_idx], [out_anchor], axis=0)
    for i in range(out_of_bounds_count):
        clip_out_idx = np.random.randint(batch_size)
        while len(in_idx_l[clip_out_idx]) == 0:
            clip_out_idx = np.random.randint(batch_size)
        clip_in_idx = np.random.randint(len(in_idx_l[clip_out_idx]))
        change_in = np.random.randint(2) == 0
        below_zero = np.random.randint(2) == 0
        change_dim_idx = np.random.randint(dim)
        if below_zero:
            (in_anchors_l if change_in else out_anchors_l)[clip_out_idx][clip_in_idx][change_dim_idx] = \
                np.int32(np.random.randint(5) - 5)
        else:
            (in_anchors_l if change_in else out_anchors_l)[clip_out_idx][clip_in_idx][change_dim_idx] = \
                np.int32(
                    (input_size if change_in else output_size)[change_dim_idx] -
                    shapes_l[clip_out_idx][clip_in_idx][change_dim_idx] +
                    np.random.randint(5) + 1
                )

    return in_idx_l, in_anchors_l, shapes_l, out_anchors_l


def get_pipeline(
        batch_size=4,
        in_size=None,
        out_size=None,
        even_paste_count=False,
        k=4,
        dtype=types.UINT8,
        no_intersections=True,
        full_input=False,
        in_anchor_top_left=False,
        in_anchor_range=None,
        out_anchor_top_left=False,
        out_anchor_range=None,
        use_gpu=False,
        num_out_of_bounds=0
):
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0, seed=np.random.randint(12345))
    with pipe:
        input, _ = fn.readers.file(file_root=img_dir)
        decoded = fn.decoders.image(input, device='cpu', output_type=types.RGB)
        resized = fn.resize(decoded, resize_x=in_size[1], resize_y=in_size[0])
        in_idx_l, in_anchors_l, shapes_l, out_anchors_l = prepare_cuts(
            k, batch_size, in_size, out_size, even_paste_count,
            no_intersections, full_input, in_anchor_top_left, in_anchor_range,
            out_anchor_top_left, out_anchor_range, num_out_of_bounds)
        in_idx = fn.external_source(lambda: in_idx_l)
        in_anchors = fn.external_source(lambda: in_anchors_l)
        shapes = fn.external_source(lambda: shapes_l)
        out_anchors = fn.external_source(lambda: out_anchors_l)
        kwargs = {
            "in_ids": in_idx,
            "output_size": out_size,
            "dtype": dtype
        }

        if not full_input:
            kwargs["shapes"] = shapes

        if not in_anchor_top_left:
            kwargs["in_anchors"] = in_anchors

        if not out_anchor_top_left:
            kwargs["out_anchors"] = out_anchors

        pasted = fn.multi_paste(resized.gpu() if use_gpu else resized, **kwargs)
        pipe.set_outputs(pasted, resized)
    return pipe, in_idx_l, in_anchors_l, shapes_l, out_anchors_l


def verify_out_of_bounds(batch_size, in_idx_l, in_anchors_l, shapes_l, out_anchors_l, in_size, out_size):
    for i in range(batch_size):
        for j, idx in enumerate(in_idx_l[i]):
            dim = len(in_anchors_l[i][j])
            for d in range(dim):
                if in_anchors_l[i][j][d] < 0 or out_anchors_l[i][j][d] < 0 or \
                        in_anchors_l[i][j][d] + shapes_l[i][j][d] > in_size[d] or \
                        out_anchors_l[i][j][d] + shapes_l[i][j][d] > out_size[d]:
                    return True
    return False



def manual_verify(batch_size, inp, output, in_idx_l, in_anchors_l, shapes_l, out_anchors_l, out_size_l, dtype):
    for i in range(batch_size):
        out = output.at(i)
        out_size = out_size_l[i]
        assert out.shape == out_size
        ref = np.zeros(out.shape)
        for j, idx in enumerate(in_idx_l[i]):
            roi_start = in_anchors_l[i][j]
            roi_end = roi_start + shapes_l[i][j]
            out_start = out_anchors_l[i][j]
            out_end = out_start + shapes_l[i][j]
            ref[out_start[0]:out_end[0], out_start[1]:out_end[1]] = inp.at(idx)[roi_start[0]:roi_end[0],
                                                                    roi_start[1]:roi_end[1]]
        ref = ref.astype(np_type_map[dtype])
        if DEBUG_LVL > 0 and not np.array_equal(out, ref):
            print(f"Error on image {i}")
            import PIL.Image
            PIL.Image.fromarray(out).save("multipaste_out.png")
            PIL.Image.fromarray(ref).save("multipaste_ref.png")
        assert np.array_equal(out, ref)


def show_images(batch_size, image_batch):
    import matplotlib.gridspec as gridspec
    import matplotlib.pyplot as plt
    columns = 4
    rows = (batch_size + 1) // (columns)
    fig = plt.figure(figsize=(32, (32 // columns) * rows))
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows * columns):
        plt.subplot(gs[j])
        plt.axis("off")
        plt.imshow(image_batch.at(j))
    plt.show()


def check_operator_multipaste(bs, pastes, in_size, out_size, even_paste_count, no_intersections, full_input, in_anchor_top_left,
                              in_anchor_range, out_anchor_top_left, out_anchor_range, out_dtype, num_out_of_bounds, device):
    pipe, in_idx_l, in_anchors_l, shapes_l, out_anchors_l = get_pipeline(
        batch_size=bs,
        in_size=in_size,
        out_size=out_size,
        even_paste_count=even_paste_count,
        k=pastes,
        dtype=out_dtype,
        no_intersections=no_intersections,
        full_input=full_input,
        in_anchor_top_left=in_anchor_top_left,
        in_anchor_range=in_anchor_range,
        out_anchor_top_left=out_anchor_top_left,
        out_anchor_range=out_anchor_range,
        num_out_of_bounds=num_out_of_bounds,
        use_gpu=device == 'gpu'
    )
    pipe.build()
    try:
        result, input = pipe.run()
        r = result.as_cpu() if device == 'gpu' else result
        if SHOW_IMAGES:
            show_images(bs, r)
        assert not verify_out_of_bounds(bs, in_idx_l, in_anchors_l, shapes_l, out_anchors_l, in_size, out_size)
        manual_verify(bs, input, r, in_idx_l, in_anchors_l, shapes_l, out_anchors_l, [out_size + (3,)] * bs, out_dtype)
    except RuntimeError as e:
        if "Paste in/out coords should be within input/output bounds" in str(e):
            assert verify_out_of_bounds(bs, in_idx_l, in_anchors_l, shapes_l, out_anchors_l, in_size, out_size)
        else:
            assert False

def test_operator_multipaste():
    tests = [
        # The arguments are:
        # - batch size
        # - average paster per output
        # - input dimensions
        # - output dimensions
        # - should each output have same number of pastes
        # - should generated pastes have no intersections
        # - should "shapes" parameter be omitted (shape to cover from input anchor to input end)
        # - should "in_anchors" parameter be omitted
        # - (Optional) in_anchor value range ((xmin, y_min), (xmax, ymax))
        # - should "out_anchors" parameter be omitted
        # - (Optional) out_anchor value range ((xmin, y_min), (xmax, ymax))
        # - output dtype
        # - number of out-of-bounds anchor changes
        [4, 2, (128, 256), (128, 128), False, False, False, False, None, False, None, types.UINT8, 0],
        [4, 2, (256, 128), (128, 128), False, True, False, False, None, False, None, types.UINT8, 0],
        [4, 2, (128, 128), (256, 128), True, False, False, False, None, False, None, types.UINT8, 0],
        [4, 2, (128, 128), (128, 256), True, True, False, False, None, False, None, types.UINT8, 0],

        [4, 2, (64, 64), (128, 128), False, False, True, False, None, False, None, types.UINT8, 0],
        [4, 2, (64, 64), (128, 128), False, False, True, False, ((10, 10), (20, 20)), False, None, types.UINT8, 0],
        [4, 2, (64, 64), (128, 128), False, False, False, True, None, False, None, types.UINT8, 0],
        [4, 2, (64, 64), (128, 128), False, False, False, False, None, True, None, types.UINT8, 0],

        [4, 2, (128, 128), (128, 128), False, False, False, False, None, False, None, types.UINT8, 0],
        [4, 2, (128, 128), (128, 128), False, False, False, False, None, False, None, types.INT16, 0],
        [4, 2, (128, 128), (128, 128), False, False, False, False, None, False, None, types.INT32, 0],
        [4, 2, (128, 128), (128, 128), False, False, False, False, None, False, None, types.FLOAT, 0],

        [4, 2, (128, 256), (128, 128), False, False, False, False, None, False, None, types.UINT8, 4],
    ]
    for t in tests:
        yield (check_operator_multipaste, *t, "cpu")
        yield (check_operator_multipaste, *t, "gpu")
