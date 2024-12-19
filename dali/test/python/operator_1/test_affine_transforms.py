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

import random
import numpy as np
from scipy.spatial.transform import Rotation as scipy_rotate

from nvidia.dali.pipeline import Pipeline

# Just to verify that import works as expected
import nvidia.dali.ops.transforms as _unused_import  # noqa:F401
import nvidia.dali.ops as ops
import nvidia.dali.fn as fn
from nvidia.dali import pipeline_def

from sequences_test_utils import ArgData, ArgDesc, sequence_suite_helper, ArgCb, ParamsProvider
from nose_utils import assert_raises, assert_warns


def check_results_sample(T1, mat_ref, T0=None, reverse=False, atol=1e-6):
    ndim = mat_ref.shape[0] - 1
    ref_T1 = None
    if T0 is not None:
        mat_T0 = np.identity(ndim + 1)
        mat_T0[:ndim, :] = T0
        if reverse:
            mat_T1 = np.dot(mat_T0, mat_ref)
        else:
            mat_T1 = np.dot(mat_ref, mat_T0)
        ref_T1 = mat_T1[:ndim, :]
    else:
        ref_T1 = mat_ref[:ndim, :]
    assert np.allclose(T1, ref_T1, atol=atol)


def check_results(T1, batch_size, mat_ref, T0=None, reverse=False, atol=1e-6):
    for idx in range(batch_size):
        check_results_sample(
            T1.at(idx), mat_ref, T0.at(idx) if T0 is not None else None, reverse, atol
        )


def translate_affine_mat(offset):
    ndim = len(offset)
    affine_mat = np.identity(ndim + 1)
    affine_mat[:ndim, -1] = offset
    return affine_mat


def check_transform_translation_op(
    offset, has_input=False, reverse_order=False, batch_size=1, num_threads=4, device_id=0
):
    ndim = len(offset)
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=1234)
    with pipe:
        if has_input:
            T0 = fn.random.uniform(range=(-1, 1), shape=(ndim, ndim + 1))
            T1 = fn.transforms.translation(
                T0, device="cpu", offset=offset, reverse_order=reverse_order
            )
            pipe.set_outputs(T1, T0)
        else:
            T1 = fn.transforms.translation(device="cpu", offset=offset)
            pipe.set_outputs(T1)
    outs = pipe.run()
    ref_mat = translate_affine_mat(offset=offset)
    T0 = outs[1] if has_input else None
    check_results(outs[0], batch_size, ref_mat, T0, reverse_order)


def test_transform_translation_op(batch_size=3, num_threads=4, device_id=0):
    for offset in [(0.0, 1.0), (2.0, 1.0, 3.0)]:
        for has_input in [False, True]:
            for reverse_order in [False, True] if has_input else [False]:
                yield (
                    check_transform_translation_op,
                    offset,
                    has_input,
                    reverse_order,
                    batch_size,
                    num_threads,
                    device_id,
                )


def scale_affine_mat(scale, center=None, ndim=None):
    if ndim is None:
        ndim = len(scale)
    else:
        assert ndim == len(scale) or 1 == len(scale)
    assert center is None or len(center) == ndim

    s_mat = np.identity(ndim + 1)
    for d in range(ndim):
        s_mat[d, d] = scale[0] if len(scale) == 1 else scale[d]

    if center is not None:
        neg_offset = [-x for x in center]
        t1_mat = translate_affine_mat(neg_offset)
        t2_mat = translate_affine_mat(center)
        affine_mat = np.dot(t2_mat, np.dot(s_mat, t1_mat))
    else:
        affine_mat = s_mat

    return affine_mat


def check_transform_scale_op(
    scale,
    center=None,
    has_input=False,
    reverse_order=False,
    ndim=None,
    batch_size=1,
    num_threads=4,
    device_id=0,
):
    if ndim is None:
        ndim = len(scale)
    assert center is None or len(center) == ndim

    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=1234)
    with pipe:
        if has_input:
            T0 = fn.random.uniform(range=(-1, 1), shape=(ndim, ndim + 1))
            T1 = fn.transforms.scale(
                T0, device="cpu", scale=scale, center=center, ndim=ndim, reverse_order=reverse_order
            )
            pipe.set_outputs(T1, T0)
        else:
            T1 = fn.transforms.scale(device="cpu", scale=scale, center=center, ndim=ndim)
            pipe.set_outputs(T1)
    outs = pipe.run()
    ref_mat = scale_affine_mat(scale=scale, center=center, ndim=ndim)
    T0 = outs[1] if has_input else None
    check_results(outs[0], batch_size, ref_mat, T0, reverse_order)


def test_transform_scale_op(batch_size=3, num_threads=4, device_id=0):
    for scale, center, ndim in [
        ((0.0, 1.0), None, None),
        ((2.0, 1.0, 3.0), None, None),
        ((2.0, 1.0), (1.0, 0.5), None),
        ((2.0,), (1.0, 0.5), 2),
    ]:
        for has_input in [False, True]:
            for reverse_order in [False, True] if has_input else [False]:
                yield (
                    check_transform_scale_op,
                    scale,
                    center,
                    has_input,
                    reverse_order,
                    ndim,
                    batch_size,
                    num_threads,
                    device_id,
                )


def rotate_affine_mat(angle, axis=None, center=None):
    assert axis is None or len(axis) == 3
    ndim = 3 if axis is not None else 2
    assert center is None or len(center) == ndim

    angle_rad = angle * np.pi / 180.0
    if ndim == 2:
        c = np.cos(angle_rad)
        s = np.sin(angle_rad)
        r_mat = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    else:  # ndim == 3
        norm_axis = axis / np.linalg.norm(axis)
        r_mat = np.identity(ndim + 1)
        r_mat[:ndim, :ndim] = scipy_rotate.from_rotvec(angle_rad * norm_axis).as_matrix()
    if center is not None:
        neg_offset = [-x for x in center]
        t1_mat = translate_affine_mat(neg_offset)
        t2_mat = translate_affine_mat(center)
        affine_mat = np.dot(t2_mat, np.dot(r_mat, t1_mat))
    else:
        affine_mat = r_mat

    return affine_mat


def check_transform_rotation_op(
    angle=None,
    axis=None,
    center=None,
    has_input=False,
    reverse_order=False,
    batch_size=1,
    num_threads=4,
    device_id=0,
):
    assert axis is None or len(axis) == 3
    ndim = 3 if axis is not None else 2
    assert center is None or len(center) == ndim
    random_angle = angle is None

    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=12345)
    with pipe:
        outputs = []
        if random_angle:
            angle = fn.random.uniform(range=(-90, 90))

        if has_input:
            T0 = fn.random.uniform(range=(-1, 1), shape=(ndim, ndim + 1))
            T1 = fn.transforms.rotation(
                T0, device="cpu", angle=angle, axis=axis, center=center, reverse_order=reverse_order
            )
            outputs = [T1, T0]
        else:
            T1 = fn.transforms.rotation(device="cpu", angle=angle, axis=axis, center=center)
            outputs = [T1]

        if random_angle:
            outputs.append(angle)

        pipe.set_outputs(*outputs)
    outs = pipe.run()
    out_idx = 1
    out_T0 = None
    out_angle = None
    if has_input:
        out_T0 = outs[out_idx]
        out_idx = out_idx + 1
    if random_angle:
        out_angle = outs[out_idx]
        out_idx = out_idx + 1
    for idx in range(batch_size):
        T0 = out_T0.at(idx) if has_input else None
        angle = out_angle.at(idx) if random_angle else angle
        ref_mat = rotate_affine_mat(angle=angle, axis=axis, center=center)
        check_results_sample(outs[0].at(idx), ref_mat, T0, reverse_order, atol=2e-6)


def test_transform_rotation_op(batch_size=3, num_threads=4, device_id=0):
    for angle, axis, center in [
        (None, None, None),
        (30.0, None, None),
        (None, None, (1.0, 0.5)),
        (30.0, None, (1.0, 0.5)),
        (40.0, (0.4, 0.3, 0.1), None),
        (40.0, (0.4, 0.3, 0.1), (1.0, -0.4, 10.0)),
        (None, (0.4, 0.3, 0.1), (1.0, -0.4, 10.0)),
    ]:
        for has_input in [False, True]:
            for reverse_order in [False, True] if has_input else [False]:
                yield (
                    check_transform_rotation_op,
                    angle,
                    axis,
                    center,
                    has_input,
                    reverse_order,
                    batch_size,
                    num_threads,
                    device_id,
                )


def shear_affine_mat(shear=None, angles=None, center=None):
    assert shear is not None or angles is not None

    if isinstance(shear, (list, tuple)):
        shear = np.float32(shear)
    if isinstance(angles, (list, tuple)):
        angles = np.float32(angles)

    if shear is None:
        shear = np.tan(angles * np.pi / 180.0)
    assert shear.size == 2 or shear.size == 6
    ndim = 3 if shear.size == 6 else 2
    assert center is None or len(center) == ndim

    if ndim == 2:
        sxy, syx = np.float32(shear).flatten()
        s_mat = np.array([[1.0, sxy, 0.0], [syx, 1.0, 0.0], [0.0, 0.0, 1.0]])
    else:  # ndim == 3
        sxy, sxz, syx, syz, szx, szy = np.float32(shear).flatten()
        s_mat = np.array([[1, sxy, sxz, 0], [syx, 1, syz, 0], [szx, szy, 1, 0], [0, 0, 0, 1]])

    if center is not None:
        neg_offset = [-x for x in center]
        t1_mat = translate_affine_mat(neg_offset)
        t2_mat = translate_affine_mat(center)
        affine_mat = np.dot(t2_mat, np.dot(s_mat, t1_mat))
    else:
        affine_mat = s_mat

    return affine_mat


def check_transform_shear_op(
    shear=None,
    angles=None,
    center=None,
    has_input=False,
    reverse_order=False,
    batch_size=1,
    num_threads=4,
    device_id=0,
):
    assert shear is not None or angles is not None
    if shear is not None:
        assert len(shear) == 2 or len(shear) == 6
        ndim = 3 if len(shear) == 6 else 2
    else:
        assert len(angles) == 2 or len(angles) == 6
        ndim = 3 if len(angles) == 6 else 2
    assert center is None or len(center) == ndim

    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=1234)
    with pipe:
        if has_input:
            T0 = fn.random.uniform(range=(-1, 1), shape=(ndim, ndim + 1))
            T1 = fn.transforms.shear(
                T0,
                device="cpu",
                shear=shear,
                angles=angles,
                center=center,
                reverse_order=reverse_order,
            )
            pipe.set_outputs(T1, T0)
        else:
            T1 = fn.transforms.shear(device="cpu", shear=shear, angles=angles, center=center)
            pipe.set_outputs(T1)
    outs = pipe.run()
    ref_mat = shear_affine_mat(shear=shear, angles=angles, center=center)
    T0 = outs[1] if has_input else None
    check_results(outs[0], batch_size, ref_mat, T0, reverse_order, atol=1e-6)


def check_transform_shear_op_runtime_args(
    ndim,
    use_angles,
    use_center,
    has_input=False,
    reverse_order=False,
    batch_size=1,
    num_threads=4,
    device_id=0,
):
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=1234)
    with pipe:
        inputs = [fn.random.uniform(range=(-1, 1), shape=(ndim, ndim + 1))] if has_input else []
        params = []
        angles_arg = None
        shear_arg = None
        center_arg = None
        if use_angles:
            angles_arg = fn.random.uniform(range=(-80, 80), shape=[ndim, ndim - 1])
            params.append(angles_arg)
        else:
            shear_arg = fn.random.uniform(range=(-2, 2), shape=[ndim, ndim - 1])
            params.append(shear_arg)
        if use_center:
            center_arg = fn.random.uniform(range=(-10, 10), shape=[ndim])
            params.append(center_arg)

        T1 = fn.transforms.shear(
            *inputs,
            device="cpu",
            shear=shear_arg,
            angles=angles_arg,
            center=center_arg,
            reverse_order=reverse_order,
        )
        pipe.set_outputs(T1, *inputs, *params)
    for _ in range(3):
        outs = pipe.run()
        T0 = outs[1] if has_input else None
        shear_param = outs[2 if has_input else 1]
        center_param = outs[3 if has_input else 2] if use_center else None
        for idx in range(batch_size):
            angles = None
            shear = None
            center = None
            if use_angles:
                angles = shear_param.at(idx)
            else:
                shear = shear_param.at(idx)
            if use_center:
                center = center_param.at(idx)
            ref_mat = shear_affine_mat(shear=shear, angles=angles, center=center)
            inp = T0.at(idx) if T0 is not None else None
            check_results_sample(outs[0].at(idx), ref_mat, inp, reverse_order, atol=1e-6)


def test_transform_shear_op(batch_size=3, num_threads=4, device_id=0):
    for shear, angles, center in [
        ((1.0, 2.0), None, None),
        ((1.0, 2.0), None, (0.4, 0.5)),
        ((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), None, None),
        ((1.0, 2.0, 3.0, 4.0, 5.0, 6.0), None, (0.4, 0.5, 0.6)),
        (None, (30.0, 10.0), None),
        (None, (30.0, 10.0), (0.4, 0.5)),
        (None, (40.0, 30.0, 10.0, 35.0, 25.0, 15.0), None),
        (None, (40.0, 30.0, 10.0, 35.0, 25.0, 15.0), (0.4, 0.5, 0.6)),
    ]:
        for has_input in [False, True]:
            for reverse_order in [False, True] if has_input else [False]:
                yield (
                    check_transform_shear_op,
                    shear,
                    angles,
                    center,
                    has_input,
                    reverse_order,
                    batch_size,
                    num_threads,
                    device_id,
                )


def test_transform_shear_op_runtime_args(batch_size=3, num_threads=4, device_id=0):
    for ndim in [2, 3]:
        for use_angles in [False, True]:
            for use_center in [False, True]:
                for has_input in [False, True]:
                    for reverse_order in [False, True] if has_input else [False]:
                        yield (
                            check_transform_shear_op_runtime_args,
                            ndim,
                            use_angles,
                            use_center,
                            has_input,
                            reverse_order,
                            4,
                            4,
                        )


def get_ndim(from_start, from_end, to_start, to_end):
    sizes = [len(a) for a in [from_start, from_end, to_start, to_end] if a is not None]
    ndim = max(sizes) if len(sizes) > 0 else 1
    for sz in sizes:
        assert sz == ndim or sz == 1
    return ndim


def expand_dims(from_start, from_end, to_start, to_end):
    ndim = get_ndim(from_start, from_end, to_start, to_end)

    def expand(arg, ndim, default_arg):
        if arg is None:
            return [default_arg] * ndim
        elif len(arg) == 1:
            return [arg[0]] * ndim
        else:
            assert len(arg) == ndim
            return arg

    return [
        expand(from_start, ndim, 0.0),
        expand(from_end, ndim, 1.0),
        expand(to_start, ndim, 0.0),
        expand(to_end, ndim, 1.0),
    ]


def crop_affine_mat(from_start, from_end, to_start, to_end, absolute=False):
    from_start, from_end, to_start, to_end = (
        np.array(x) for x in expand_dims(from_start, from_end, to_start, to_end)
    )
    if absolute:
        from_start, from_end = np.minimum(from_start, from_end), np.maximum(from_start, from_end)
        to_start, to_end = np.minimum(to_start, to_end), np.maximum(to_start, to_end)

    scale = (to_end - to_start) / (from_end - from_start)
    T1 = translate_affine_mat(-from_start)
    S = scale_affine_mat(scale)
    T2 = translate_affine_mat(to_start)
    affine_mat = np.dot(T2, np.dot(S, T1))
    return affine_mat


def check_transform_crop_op(
    from_start=None,
    from_end=None,
    to_start=None,
    to_end=None,
    absolute=False,
    has_input=False,
    reverse_order=False,
    batch_size=1,
    num_threads=4,
    device_id=0,
):
    ndim = get_ndim(from_start, from_end, to_start, to_end)
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id, seed=1234)
    with pipe:
        if has_input:
            T0 = fn.random.uniform(range=(-1, 1), shape=(ndim, ndim + 1))
            T1 = fn.transforms.crop(
                T0,
                device="cpu",
                from_start=from_start,
                from_end=from_end,
                to_start=to_start,
                to_end=to_end,
                absolute=absolute,
                reverse_order=reverse_order,
            )
            pipe.set_outputs(T1, T0)
        else:
            T1 = fn.transforms.crop(
                device="cpu",
                from_start=from_start,
                from_end=from_end,
                to_start=to_start,
                to_end=to_end,
                absolute=absolute,
            )
            pipe.set_outputs(T1)
    outs = pipe.run()

    ref_mat = crop_affine_mat(from_start, from_end, to_start, to_end, absolute=absolute)
    T0 = outs[1] if has_input else None
    T1 = outs[0]
    check_results(T1, batch_size, ref_mat, T0, reverse_order, atol=1e-6)
    if not has_input:
        from_start, from_end, to_start, to_end = expand_dims(from_start, from_end, to_start, to_end)
        if absolute:
            from_start, from_end = np.minimum(from_start, from_end), np.maximum(
                from_start, from_end
            )
            to_start, to_end = np.minimum(to_start, to_end), np.maximum(to_start, to_end)
        for idx in range(batch_size):
            MT = T1.at(idx)
            M, T = MT[:ndim, :ndim], MT[:, ndim]
            assert np.allclose(np.dot(M, from_start) + T, to_start, atol=1e-6)
            assert np.allclose(np.dot(M, from_end) + T, to_end, atol=1e-6)


def test_transform_crop_op(batch_size=3, num_threads=4, device_id=0):
    for from_start, from_end, to_start, to_end in [
        (None, None, None, None),
        ((0.1, 0.2), (1.0, 1.2), (0.3, 0.2), (0.5, 0.6)),
        ((0.1, 0.2), (0.4, 0.9), None, None),
        ((0.2, 0.2), None, None, None),
        (None, (0.4, 0.9), None, None),
        ((0.1, 0.2, 0.3), (1.0, 1.2, 1.3), (0.3, 0.2, 0.1), (0.5, 0.6, 0.7)),
        ((0.1, 0.2, 0.3), (1.0, 1.2, 1.3), None, None),
    ]:
        for has_input in [False, True]:
            for reverse_order in [False, True] if has_input else [False]:
                yield (
                    check_transform_crop_op,
                    from_start,
                    from_end,
                    to_start,
                    to_end,
                    False,
                    has_input,
                    reverse_order,
                    batch_size,
                    num_threads,
                    device_id,
                )
                # Reversed start and end
                for absolute in [False, True]:
                    yield (
                        check_transform_crop_op,
                        from_end,
                        from_start,
                        to_end,
                        to_start,
                        absolute,
                        has_input,
                        reverse_order,
                        batch_size,
                        num_threads,
                        device_id,
                    )


def check_combine_transforms(
    num_transforms=2, ndim=2, reverse_order=False, batch_size=1, num_threads=4, device_id=0
):
    pipe = Pipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
    with pipe:
        transforms = [
            fn.random.uniform(range=(-1, 1), shape=(ndim, ndim + 1), seed=1234)
            for _ in range(num_transforms)
        ]
        T = fn.transforms.combine(*transforms)
    pipe.set_outputs(T, *transforms)
    outs = pipe.run()
    for idx in range(batch_size):
        num_mats = len(outs) - 1
        assert num_mats >= 2
        mats = [np.identity(ndim + 1) for _ in range(num_mats)]
        for in_idx in range(len(mats)):
            mats[in_idx][:ndim, :] = outs[1 + in_idx].at(idx)

        # by default we want to access them in opposite order
        if not reverse_order:
            mats.reverse()
        ref_mat = np.identity(ndim + 1)
        for mat in mats:
            ref_mat = np.dot(mat, ref_mat)

        assert np.allclose(outs[0].at(idx), ref_mat[:ndim, :], atol=1e-6)


def test_combine_transforms(batch_size=3, num_threads=4, device_id=0):
    for num_transforms in [2, 3, 10]:
        for ndim in [2, 3, 6]:
            for reverse_order in [False, True]:
                yield (
                    check_combine_transforms,
                    num_transforms,
                    ndim,
                    reverse_order,
                    batch_size,
                    num_threads,
                    device_id,
                )


def test_combine_transforms_correct_order():
    batch_size = 3
    pipe = Pipeline(batch_size=batch_size, num_threads=4, device_id=0)
    with pipe:
        import nvidia.dali.fn.transforms as T

        t1 = T.translation(offset=(1, 2))
        t2 = T.rotation(angle=30.0)
        t12 = T.rotation(T.translation(offset=(1, 2)), angle=30.0)
        t21 = T.translation(T.rotation(angle=30.0), offset=(1, 2))
        pipe.set_outputs(T.combine(t1, t2), t12, T.combine(t1, t2, reverse_order=True), t21)
    outs = pipe.run()
    for idx in range(batch_size):
        assert np.allclose(outs[0].at(idx), outs[1].at(idx), atol=1e-6)
        assert np.allclose(outs[2].at(idx), outs[3].at(idx), atol=1e-6)


def test_transform_translation_deprecation():
    fmt = (
        "WARNING: `nvidia.dali.{}` is now deprecated."
        " Use `nvidia.dali.fn.transforms.translation` instead."
    )
    with assert_warns(DeprecationWarning, glob=fmt.format("fn.transform_translation")):
        fn.transform_translation(offset=(0, 0))
    with assert_warns(DeprecationWarning, glob=fmt.format("ops.TransformTranslation")):
        ops.TransformTranslation(offset=(0, 0))()


def test_sequences():
    np_rng = np.random.default_rng(12345)
    rng = random.Random(42)
    num_iters = 4
    max_num_frames = 50
    max_batch_size = 12

    class TransformsParamsProvider(ParamsProvider):
        def unfold_output_layout(self, layout):
            unfolded = super().unfold_output_layout(layout)
            if unfolded == "**":
                return ""
            return unfolded

    def rand_range(limit):
        return range(rng.randint(1, limit) + 1)

    def mt(desc):
        return np.float32(np_rng.uniform(-20, 20, (2, 3)))

    def scale(desc):
        return np.array([rng.randint(0, 5), rng.randint(-50, 20)], dtype=np.float32)

    def shift(desc):
        return np.array([rng.randint(-100, 200), rng.randint(-50, 20)], dtype=np.float32)

    def shear_angles(desc):
        return np.array([rng.randint(-90, 90), rng.randint(-90, 90)], dtype=np.float32)

    def angle(desc):
        return np.array(rng.uniform(-180, 180), dtype=np.float32)

    def per_frame_input(frame_cb):
        return [
            [
                np.array([frame_cb(None) for _ in rand_range(max_num_frames)], dtype=np.float32)
                for _ in rand_range(max_batch_size)
            ]
            for _ in range(num_iters)
        ]

    test_cases = [
        (
            fn.transforms.rotation,
            {},
            TransformsParamsProvider([ArgCb("angle", angle, True)]),
            ["cpu"],
        ),
        (
            fn.transforms.rotation,
            {"reverse_order": True},
            TransformsParamsProvider([ArgCb("center", shift, True), ArgCb("angle", angle, False)]),
            ["cpu"],
        ),
        (
            fn.transforms.scale,
            {},
            TransformsParamsProvider([ArgCb("scale", scale, True), ArgCb("center", shift, False)]),
            ["cpu"],
        ),
        (
            fn.transforms.scale,
            {"center": np.array([-50, 100], dtype=np.float32)},
            TransformsParamsProvider([ArgCb("scale", scale, True)]),
            ["cpu"],
        ),
        (
            fn.transforms.translation,
            {},
            TransformsParamsProvider([ArgCb("offset", shift, True)]),
            ["cpu"],
        ),
        (
            fn.transforms.shear,
            {},
            TransformsParamsProvider([ArgCb("angles", shear_angles, True)]),
            ["cpu"],
        ),
        (fn.transforms.shear, {}, TransformsParamsProvider([ArgCb("shear", shift, True)]), ["cpu"]),
        (
            fn.transforms.combine,
            {},
            TransformsParamsProvider([ArgCb(2, mt, True), ArgCb(1, mt, False)]),
            ["cpu"],
        ),
        (fn.transforms.combine, {}, TransformsParamsProvider([ArgCb(1, mt, True)]), ["cpu"]),
    ]
    only_with_seq_input_cases = [
        (
            fn.transforms.combine,
            {},
            TransformsParamsProvider([ArgCb(1, mt, True), ArgCb(2, mt, True)]),
            ["cpu"],
        ),
        (fn.transforms.combine, {}, TransformsParamsProvider([ArgCb(1, mt, False)]), ["cpu"]),
        (
            fn.transforms.translation,
            {},
            TransformsParamsProvider([ArgCb("offset", shift, False)]),
            ["cpu"],
        ),
        (
            fn.transforms.rotation,
            {"reverse_order": True, "angle": 92.0},
            TransformsParamsProvider([]),
            ["cpu"],
        ),
    ]

    seq_cases = test_cases + only_with_seq_input_cases
    main_input = ArgData(desc=ArgDesc(0, "F", "", "F**"), data=per_frame_input(mt))
    yield from sequence_suite_helper(rng, [main_input], seq_cases, num_iters)

    # transform the test cases to test the transforms with per-frame args but:
    # 1. with the positional input that does not contain frames
    # 2. without the positional input
    for tested_fn, fixed_params, params_provider, devices in test_cases:
        [main_source, *rest_cbs] = params_provider.input_params
        if main_source.desc.expandable_prefix != "F":
            continue
        broadcast_0_pos_case_params = TransformsParamsProvider([ArgCb(0, mt, False), *rest_cbs])
        broadcast_0_pos_case = (tested_fn, fixed_params, broadcast_0_pos_case_params, devices)
        if any(source.desc.is_positional_arg for source in params_provider.input_params):
            cases = [broadcast_0_pos_case]
        else:
            no_pos_case_params = TransformsParamsProvider(rest_cbs)
            no_pos_input_case = (tested_fn, fixed_params, no_pos_case_params, devices)
            cases = [broadcast_0_pos_case, no_pos_input_case]
        per_frame_data = per_frame_input(main_source.cb)
        data_dim = len(per_frame_data[0][0].shape)
        assert data_dim > 0
        data_layout = "F" + "*" * (data_dim - 1)
        main_input = ArgData(
            desc=ArgDesc(main_source.desc.name, "F", "", data_layout), data=per_frame_data
        )
        yield from sequence_suite_helper(rng, [main_input], cases, num_iters)


def test_combine_shape_mismatch():
    np_rng = np.random.default_rng(42)
    batch_size = 8
    num_frames = 50

    def mt():
        return np.float32(np_rng.uniform(-100, 250, (num_frames, 2, 3)))

    batch0_inp = [mt() for _ in range(batch_size)]
    batch1_inp = [mt()[i:] for i in range(batch_size)]

    expected_msg = "The input 0 and the input 1 have different number of frames for sample 1"
    with assert_raises(RuntimeError, glob=expected_msg):

        @pipeline_def
        def pipeline():
            mts0, mts1 = fn.external_source(lambda _: (batch0_inp, batch1_inp), num_outputs=2)
            return fn.transforms.combine(fn.per_frame(mts0), fn.per_frame(mts1))

        pipe = pipeline(batch_size=batch_size, num_threads=4, device_id=0)
        pipe.run()


def test_rotate_shape_mismatch():
    np_rng = np.random.default_rng(42)
    batch_size = 8
    num_frames = 50

    def mt():
        return np.float32(np_rng.uniform(-100, 250, (num_frames, 2, 3)))

    mts_inp = [mt() for _ in range(batch_size)]
    angles_inp = [
        np.array([angle for angle in range(num_frames - i)], dtype=np.float32)
        for i in range(batch_size)
    ]
    centers_inp = [
        np.array([c for c in range(num_frames + i)], dtype=np.float32) for i in range(batch_size)
    ]

    with assert_raises(
        RuntimeError,
        glob="The sample 1 of tensor argument `angle` "
        "contains 49 per-frame parameters, but there are 50 frames in "
        "the corresponding sample of input 0.",
    ):

        @pipeline_def
        def pipeline():
            mts, angles = fn.external_source(lambda _: (mts_inp, angles_inp), num_outputs=2)
            return fn.transforms.rotation(fn.per_frame(mts), angle=fn.per_frame(angles))

        pipe = pipeline(batch_size=batch_size, num_threads=4, device_id=0)
        pipe.run()

    with assert_raises(
        RuntimeError,
        glob="The sample 1 of tensor argument `center` contains 51 "
        "per-frame parameters, but there are 49 frames in the corresponding sample "
        "of argument `angle`",
    ):

        @pipeline_def
        def pipeline():
            angles, centers = fn.external_source(lambda _: (angles_inp, centers_inp), num_outputs=2)
            return fn.transforms.rotation(angle=fn.per_frame(angles), center=fn.per_frame(centers))

        pipe = pipeline(batch_size=batch_size, num_threads=4, device_id=0)
        pipe.run()
