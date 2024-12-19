# Copyright (c) 2021-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
from nvidia.dali import fn, pipeline_def, types
import numpy as np
import os
from test_utils import get_dali_extra_path
from nose_utils import assert_raises
from nose2.tools import params, cartesian_params


DEBUG_LVL = 0
SHOW_IMAGES = False

np.random.seed(1234)

data_root = get_dali_extra_path()
img_dir = os.path.join(data_root, "db", "single", "jpeg")


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
    assert out_of_bounds_count == 0 or not no_intersections

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
                shape = (
                    [
                        np.int32(
                            np.random.randint(
                                min(input_size[i], output_size[i])
                                // (iters if no_intersections else 1)
                            )
                            + 1
                        )
                        for i in range(dim)
                    ]
                    if not full_input
                    else input_size
                )

                if in_anchor_top_left:
                    in_anchor = [0] * dim
                elif in_anchor_range is not None:
                    in_anchor = [
                        np.int32(np.random.randint(in_anchor_range[0][i], in_anchor_range[1][i]))
                        for i in range(dim)
                    ]
                    if full_input:
                        shape = [np.int32(input_size[i] - in_anchor[i]) for i in range(dim)]
                else:
                    in_anchor = [
                        np.int32(np.random.randint(input_size[i] - shape[i] + 1))
                        for i in range(dim)
                    ]

                if out_anchor_top_left:
                    out_anchor = [0] * dim
                elif out_anchor_range is not None:
                    out_anchor = [
                        np.int32(np.random.randint(out_anchor_range[0][i], out_anchor_range[1][i]))
                        for i in range(dim)
                    ]
                else:
                    out_anchor = [
                        np.int32(np.random.randint(output_size[i] - shape[i] + 1))
                        for i in range(dim)
                    ]

                if no_intersections:
                    is_ok = True
                    for k in range(len(in_idx_l[out_idx])):
                        if intersects(
                            out_anchors_l[out_idx][k], shapes_l[out_idx][k], out_anchor, shape
                        ):
                            is_ok = False
                            break
                    if not is_ok:
                        continue
                    break
                break

            if DEBUG_LVL >= 1:
                print(
                    f"""in_idx: {in_idx}, out_idx: {out_idx}, in_anchor: {
                in_anchor}, in_shape: {shape}, out_anchor: {out_anchor}"""
                )

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
            anchors = in_anchors_l if change_in else out_anchors_l
            anchors[clip_out_idx][clip_in_idx][change_dim_idx] = np.int32(np.random.randint(-5, 0))
        else:
            anchors = in_anchors_l if change_in else out_anchors_l
            size = input_size if change_in else output_size
            anchors[clip_out_idx][clip_in_idx][change_dim_idx] = np.int32(
                size[change_dim_idx]
                - shapes_l[clip_out_idx][clip_in_idx][change_dim_idx]
                + np.random.randint(5)
                + 1
            )

    return in_idx_l, in_anchors_l, shapes_l, out_anchors_l


def get_pipeline(
    batch_size=4,
    in_size=None,
    out_size=None,
    even_paste_count=False,
    use_positional=False,
    k=4,
    dtype=types.UINT8,
    no_intersections=True,
    full_input=False,
    in_anchor_top_left=False,
    in_anchor_range=None,
    out_anchor_top_left=False,
    out_anchor_range=None,
    use_gpu=False,
    num_out_of_bounds=0,
    use_shapes_rel=False,
    use_in_anchors_rel=False,
    use_out_anchors_rel=False,
):
    pipe = Pipeline(
        batch_size=batch_size, num_threads=4, device_id=0, seed=np.random.randint(12345)
    )
    with pipe:
        input, _ = fn.readers.file(file_root=img_dir)
        decoded = fn.decoders.image(input, device="cpu", output_type=types.RGB)
        resized_cpu = fn.resize(decoded, resize_x=in_size[1], resize_y=in_size[0])
        resized = resized_cpu.gpu() if use_gpu else resized_cpu
        in_idx_l, in_anchors_l, shapes_l, out_anchors_l = prepare_cuts(
            k,
            batch_size,
            in_size,
            out_size,
            even_paste_count,
            no_intersections,
            full_input,
            in_anchor_top_left,
            in_anchor_range,
            out_anchor_top_left,
            out_anchor_range,
            num_out_of_bounds,
        )
        in_idx = fn.external_source(lambda: in_idx_l)
        in_anchors = fn.external_source(lambda: in_anchors_l)
        shapes = fn.external_source(lambda: shapes_l)
        out_anchors = fn.external_source(lambda: out_anchors_l)
        kwargs = {"output_size": out_size, "dtype": dtype}
        if not use_positional:
            kwargs["in_ids"] = in_idx
            args = (resized,)
        else:
            assert even_paste_count
            assert len(in_idx_l) == batch_size

            def source(perm):
                def cb():
                    return perm

                return cb

            paste_count = len(in_idx_l[0])
            perms = [[sample[i] for sample in in_idx_l] for i in range(paste_count)]
            perms = [fn.external_source(source(perm)) for perm in perms]
            args = tuple(fn.permute_batch(resized, indices=perm) for perm in perms)

        if not full_input:
            if not use_shapes_rel:
                kwargs["shapes"] = shapes
            else:
                kwargs["shapes_rel"] = shapes / in_size

        if not in_anchor_top_left:
            if not use_in_anchors_rel:
                kwargs["in_anchors"] = in_anchors
            else:
                kwargs["in_anchors_rel"] = in_anchors / in_size

        if not out_anchor_top_left:
            if not use_out_anchors_rel:
                kwargs["out_anchors"] = out_anchors
            else:
                kwargs["out_anchors_rel"] = out_anchors / out_size

        pasted = fn.multi_paste(*args, **kwargs)
        pipe.set_outputs(pasted, resized_cpu)
    return pipe, in_idx_l, in_anchors_l, shapes_l, out_anchors_l


def verify_out_of_bounds(
    batch_size, in_idx_l, in_anchors_l, shapes_l, out_anchors_l, in_size, out_size
):
    for i in range(batch_size):
        for j, idx in enumerate(in_idx_l[i]):
            dim = len(in_anchors_l[i][j])
            for d in range(dim):
                if (
                    in_anchors_l[i][j][d] < 0
                    or out_anchors_l[i][j][d] < 0
                    or in_anchors_l[i][j][d] + shapes_l[i][j][d] > in_size[d]
                    or out_anchors_l[i][j][d] + shapes_l[i][j][d] > out_size[d]
                ):
                    return True
    return False


def manual_verify(
    batch_size, inp, output, in_idx_l, in_anchors_l, shapes_l, out_anchors_l, out_size_l, dtype
):
    for i in range(batch_size):
        ref_source_info = ";".join([inp[idx].source_info() for idx in in_idx_l[i]])
        assert (
            output[i].source_info() == ref_source_info
        ), f"{output[i].source_info()} == {ref_source_info}"
        out = output.at(i)
        out_size = out_size_l[i]
        assert out.shape == out_size, f"{out.shape} vs {out_size}"
        ref = np.zeros(out.shape)
        for j, idx in enumerate(in_idx_l[i]):
            roi_start = in_anchors_l[i][j]
            roi_end = roi_start + shapes_l[i][j]
            out_start = out_anchors_l[i][j]
            out_end = out_start + shapes_l[i][j]
            ref[out_start[0] : out_end[0], out_start[1] : out_end[1]] = inp.at(idx)[
                roi_start[0] : roi_end[0], roi_start[1] : roi_end[1]
            ]
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
    gs = gridspec.GridSpec(rows, columns)
    for j in range(rows * columns):
        plt.subplot(gs[j])
        plt.axis("off")
        plt.imshow(image_batch.at(j))
    plt.show()


def check_operator_multipaste(
    bs,
    pastes,
    in_size,
    out_size,
    even_paste_count,
    use_positional,
    no_intersections,
    full_input,
    in_anchor_top_left,
    in_anchor_range,
    out_anchor_top_left,
    out_anchor_range,
    out_dtype,
    num_out_of_bounds,
    device,
    use_shapes_rel,
    use_in_anchors_rel,
    use_out_anchors_rel,
):
    pipe, in_idx_l, in_anchors_l, shapes_l, out_anchors_l = get_pipeline(
        batch_size=bs,
        in_size=in_size,
        out_size=out_size,
        even_paste_count=even_paste_count,
        use_positional=use_positional,
        k=pastes,
        dtype=out_dtype,
        no_intersections=no_intersections,
        full_input=full_input,
        in_anchor_top_left=in_anchor_top_left,
        in_anchor_range=in_anchor_range,
        out_anchor_top_left=out_anchor_top_left,
        out_anchor_range=out_anchor_range,
        num_out_of_bounds=num_out_of_bounds,
        use_gpu=device == "gpu",
        use_shapes_rel=use_shapes_rel,
        use_in_anchors_rel=use_in_anchors_rel,
        use_out_anchors_rel=use_out_anchors_rel,
    )
    try:
        result, input = pipe.run()
        r = result.as_cpu() if device == "gpu" else result
        if SHOW_IMAGES:
            show_images(bs, r)
        assert not verify_out_of_bounds(
            bs, in_idx_l, in_anchors_l, shapes_l, out_anchors_l, in_size, out_size
        )
        manual_verify(
            bs,
            input,
            r,
            in_idx_l,
            in_anchors_l,
            shapes_l,
            out_anchors_l,
            [out_size + (3,)] * bs,
            out_dtype,
        )
    except RuntimeError as e:
        use_rel = use_shapes_rel or use_in_anchors_rel or use_out_anchors_rel

        if "The pasted region must be within" in str(e) or (
            use_rel and "values must be floats in [0, 1] range," in str(e)
        ):
            assert verify_out_of_bounds(
                bs, in_idx_l, in_anchors_l, shapes_l, out_anchors_l, in_size, out_size
            )
        else:
            assert False


def test_operator_multipaste():
    in_anchor = ((10, 10), (20, 20))
    rng = np.random.default_rng(42)
    tests = [
        # The arguments are:
        # - batch size
        # - average paster per output
        # - input dimensions
        # - output dimensions
        # - should each output have same number of pastes
        # - should the inputs be passed as positional argument
        # - should generated pastes have no intersections
        # - should "shapes" parameter be omitted (shape to cover from input anchor to input end)
        # - should "in_anchors" parameter be omitted
        # - (Optional) in_anchor value range ((xmin, y_min), (xmax, ymax))
        # - should "out_anchors" parameter be omitted
        # - (Optional) out_anchor value range ((xmin, y_min), (xmax, ymax))
        # - output dtype
        # - number of out-of-bounds anchor changes
        [
            4,
            2,
            (128, 256),
            (128, 128),
            False,
            False,
            False,
            False,
            False,
            None,
            False,
            None,
            types.UINT8,
            0,
        ],
        [
            4,
            2,
            (256, 128),
            (128, 128),
            False,
            False,
            True,
            False,
            False,
            None,
            False,
            None,
            types.UINT8,
            0,
        ],
        [
            8,
            15,
            (256, 128),
            (128, 128),
            True,
            True,
            True,
            False,
            False,
            None,
            False,
            None,
            types.UINT8,
            0,
        ],
        [
            4,
            2,
            (128, 128),
            (256, 128),
            True,
            False,
            False,
            False,
            False,
            None,
            False,
            None,
            types.UINT8,
            0,
        ],
        [
            4,
            2,
            (128, 128),
            (256, 128),
            True,
            True,
            False,
            False,
            False,
            None,
            False,
            None,
            types.UINT8,
            0,
        ],
        [
            4,
            2,
            (128, 128),
            (128, 256),
            True,
            False,
            True,
            False,
            False,
            None,
            False,
            None,
            types.UINT8,
            0,
        ],
        [
            4,
            2,
            (128, 128),
            (128, 256),
            True,
            True,
            True,
            False,
            False,
            None,
            False,
            None,
            types.UINT8,
            0,
        ],
        [
            4,
            2,
            (64, 64),
            (128, 128),
            False,
            False,
            False,
            True,
            False,
            None,
            False,
            None,
            types.UINT8,
            0,
        ],
        [
            4,
            2,
            (64, 64),
            (128, 128),
            False,
            False,
            False,
            True,
            False,
            in_anchor,
            False,
            None,
            types.UINT8,
            0,
        ],
        [
            4,
            2,
            (64, 64),
            (128, 128),
            False,
            False,
            False,
            False,
            True,
            None,
            False,
            None,
            types.UINT8,
            0,
        ],
        [
            4,
            2,
            (64, 64),
            (128, 128),
            False,
            False,
            False,
            False,
            False,
            None,
            True,
            None,
            types.UINT8,
            0,
        ],
        [
            4,
            2,
            (128, 128),
            (128, 128),
            False,
            False,
            False,
            False,
            False,
            None,
            False,
            None,
            types.UINT8,
            0,
        ],
        [
            4,
            2,
            (128, 128),
            (128, 128),
            False,
            False,
            False,
            False,
            False,
            None,
            False,
            None,
            types.INT16,
            0,
        ],
        [
            4,
            2,
            (128, 128),
            (128, 128),
            False,
            False,
            False,
            False,
            False,
            None,
            False,
            None,
            types.INT32,
            0,
        ],
        [
            4,
            2,
            (128, 128),
            (128, 128),
            True,
            True,
            False,
            False,
            False,
            None,
            False,
            None,
            types.INT32,
            0,
        ],
        [
            4,
            2,
            (128, 128),
            (128, 128),
            False,
            False,
            False,
            False,
            False,
            None,
            False,
            None,
            types.FLOAT,
            0,
        ],
        [
            4,
            2,
            (128, 128),
            (128, 128),
            True,
            True,
            False,
            False,
            False,
            None,
            False,
            None,
            types.FLOAT,
            0,
        ],
        [
            4,
            2,
            (128, 256),
            (128, 128),
            False,
            False,
            False,
            False,
            False,
            None,
            False,
            None,
            types.UINT8,
            4,
        ],
    ]
    for t in tests:
        use_rel = tuple(bool(s) for s in rng.choice(2, size=3))
        yield (check_operator_multipaste, *t, "cpu", *use_rel)
        yield (check_operator_multipaste, *t, "gpu", *use_rel)


@cartesian_params(("cpu", "gpu"), (True, False), (True, False), (None, (501, 501)))
def test_input_shape_inference(device, use_uniform_shape, use_in_idx, output_size):

    def get_image(sample_info):
        if use_uniform_shape or sample_info.idx_in_batch % 2 == 0:
            shape = (401, 501, 3)
        else:
            shape = (301, 301, 3)
        return np.full(shape, sample_info.idx_in_batch, dtype=np.uint8)

    def get_indices(sample_info):
        idx_in_batch = sample_info.idx_in_batch
        return np.array([idx_in_batch // 2, idx_in_batch // 2 + 1], dtype=np.int32)

    batch_size = 8

    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=4)
    def pipeline():
        image = fn.external_source(get_image, batch=False)
        assert device in ("gpu", "cpu")
        image = image.gpu() if device == "gpu" else image
        if use_in_idx:
            args = tuple()
            kwargs = {
                "in_ids": fn.external_source(get_indices, batch=False),
                "output_size": output_size,
            }
        else:
            image = fn.permute_batch(image, indices=fn.batch_permutation())
            args = (image,)
            kwargs = {"output_size": output_size}

        return fn.multi_paste(image, *args, **kwargs)

    p = pipeline()
    if not use_uniform_shape and output_size is None:
        with assert_raises(
            RuntimeError,
            glob="If the `output_size` is not specified, all the input "
            "samples must have the same shape.",
        ):
            p.run()
    else:
        expected_size = (output_size or (401, 501)) + (3,)
        (out,) = p.run()
        assert len(out) == batch_size
        for sample in out:
            assert tuple(sample.shape()) == expected_size, f"{sample.shape()} vs {expected_size}"


@params((-1, 8, "cpu"), (-1, 8, "gpu"), (1, 1, "cpu"), (9, 8, "gpu"))
def test_out_of_bounds_idx(out_of_bound_idx, batch_size, device):
    rng = np.random.default_rng(42)
    bogus_idx = rng.choice(list(range(batch_size)))

    def get_image(sample_info):
        return np.full((101, 101, 3), sample_info.idx_in_batch, dtype=np.uint8)

    def get_indices(sample_info):
        idx_in_batch = sample_info.idx_in_batch
        indices = np.array([idx_in_batch // 2, idx_in_batch // 2 + 1], dtype=np.int32)
        if idx_in_batch == bogus_idx:
            indices[rng.choice([0, 1])] = out_of_bound_idx
        return indices

    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=4)
    def pipeline():
        image = fn.external_source(get_image, batch=False)
        in_ids = fn.external_source(get_indices, batch=False)
        assert device in ("gpu", "cpu")
        image = image.gpu() if device == "gpu" else image
        return fn.multi_paste(image, in_ids=in_ids)

    p = pipeline()
    with assert_raises(
        RuntimeError,
        glob=f"The `in_idx` must be in range * Got in_idx: {out_of_bound_idx}. "
        f"Input batch size is: {batch_size}.",
    ):
        p.run()


@cartesian_params(("cpu", "gpu"), (True, False))
def test_conflicting_channels(device, use_positional):
    batch_size = 16

    def shift_by_one(sample_info):
        return np.array((sample_info.idx_in_batch + 1) % batch_size, dtype=np.int32)

    def get_image(sample_info):
        idx_in_batch = sample_info.idx_in_batch
        num_channels = (idx_in_batch % 4) + 1
        return np.full((101, 101, num_channels), sample_info.idx_in_batch, dtype=np.uint8)

    def get_indices(sample_info):
        idx_in_batch = sample_info.idx_in_batch
        indices = np.array([idx_in_batch // 4 + i for i in range(4)], dtype=np.int32)
        return indices

    @pipeline_def(batch_size=batch_size, device_id=0, num_threads=4)
    def pipeline():
        image = fn.external_source(get_image, batch=False)
        assert device in ("gpu", "cpu")
        image = image.gpu() if device == "gpu" else image
        if not use_positional:
            in_ids = fn.external_source(get_indices, batch=False)
            args = tuple()
            kwargs = {"in_ids": in_ids}
        else:
            perm = fn.external_source(shift_by_one, batch=False)
            args = []
            kwargs = {}
            perm_image = image
            for _ in range(3):
                perm_image = fn.permute_batch(perm_image, indices=perm)
                args.append(perm_image)
            args = tuple(args)

        return fn.multi_paste(image, *args, **kwargs)

    p = pipeline()
    with assert_raises(
        RuntimeError,
        glob="All regions pasted into given output sample must have the same number of channels",
    ):
        p.run()


@cartesian_params(("cpu", "gpu"), (True, False))
def test_var_channels(device, use_positional):
    num_regions = 9
    max_num_channels = 4
    batch_size = num_regions * max_num_channels
    num_iters = 2

    anchors = np.array(
        [
            [0, 0],
            [0, 0.5],
            [0.5, 0],
            [0.5, 0.5],
            [0.25, 0.25],
            [0, 0],
            [0, 0.75],
            [0.75, 0],
            [0.75, 0.75],
        ],
        dtype=np.float32,
    )
    shapes = np.array(
        [
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.5, 0.5],
            [0.25, 0.25],
            [0.25, 0.25],
            [0.25, 0.25],
            [0.25, 0.25],
        ],
        dtype=np.float32,
    )

    def in_ids_cb(sample_info):
        idx = sample_info.idx_in_batch
        # group every consecutive num_regions samples
        group_idx = idx // num_regions
        offset = idx % num_regions
        in_ids = [(offset + i) % num_regions + group_idx * num_regions for i in range(num_regions)]
        return np.array(in_ids, dtype=np.int32)

    def num_channels_cb(sample_info):
        idx = sample_info.idx_in_batch
        # group every consecutive num_regions samples
        group_idx = idx // num_regions
        return np.int32(group_idx + 1)

    def out_size_cb(sample_info):
        shapes = [(301, 273), (100, 200), (333, 555), (555, 333)]
        idx = sample_info.idx_in_batch
        # group every consecutive num_regions samples
        group_idx = idx // num_regions
        return np.array(shapes[group_idx], dtype=np.int32)

    @pipeline_def(
        batch_size=batch_size, device_id=0, num_threads=4, seed=42, enable_conditionals=True
    )
    def pipeline():
        in_out_size = fn.external_source(out_size_cb, batch=False)
        alpha = fn.random.uniform(
            range=[0, 255],
            shape=fn.cat(in_out_size, np.array([1], dtype=np.int32)),
            dtype=types.DALIDataType.UINT8,
        )
        encoded, _ = fn.readers.file(file_root=img_dir)
        image_base = fn.decoders.image(encoded, device="cpu", output_type=types.RGB)
        image_base = fn.resize(image_base, size=fn.cast(in_out_size, dtype=types.FLOAT))
        num_channels = fn.external_source(num_channels_cb, batch=False)
        if num_channels > 3:
            image_base = fn.cat(image_base, alpha, axis=-1)
        image_base = image_base[:, :, :num_channels]
        image = image_base.gpu() if device == "gpu" else image_base
        in_ids = fn.external_source(in_ids_cb, batch=False)
        if not use_positional:
            kwargs = {"in_ids": in_ids}
            args = (image,)
        else:
            kwargs = {}
            args = tuple(fn.permute_batch(image, indices=in_ids[i]) for i in range(num_regions))
        output = fn.multi_paste(
            *args,
            **kwargs,
            in_anchors_rel=anchors,
            out_anchors_rel=anchors,
            shapes_rel=shapes,
            output_size=in_out_size,
        )
        return output, image_base, in_ids, in_out_size, num_channels

    p = pipeline()
    for _ in range(num_iters):
        out_images, inp_images, idxs, out_sizes, num_channels = p.run()
        idxs = [np.array(sample) for sample in idxs]
        out_images = out_images.as_cpu() if device == "gpu" else out_images
        num_channels = [np.array(num_channel) for num_channel in num_channels]
        out_sizes = [np.array(size) for size in out_sizes]
        anchors_abs = [anchors] * batch_size
        assert len(anchors_abs) == len(out_sizes)
        anchors_abs = [
            np.int32(np.floor(anchor * size)) for anchor, size in zip(anchors_abs, out_sizes)
        ]
        region_shapes_abs = [shapes] * batch_size
        assert len(region_shapes_abs) == len(out_sizes)
        region_shapes_abs = [
            np.int32(np.ceil(region_shape * size))
            for region_shape, size in zip(region_shapes_abs, out_sizes)
        ]
        assert len(out_sizes) == len(num_channels)
        out_sizes = [
            tuple(size) + (num_channel.item(),)
            for size, num_channel in zip(out_sizes, num_channels)
        ]
        manual_verify(
            batch_size,
            inp_images,
            out_images,
            idxs,
            anchors_abs,
            region_shapes_abs,
            anchors_abs,
            out_sizes,
            types.UINT8,
        )
