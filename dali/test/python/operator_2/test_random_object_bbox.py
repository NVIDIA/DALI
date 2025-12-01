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


# nose_utils goes first to deal with Python 3.10 incompatibility
from nose_utils import attr, nottest, assert_raises
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import numpy as np
import scipy.ndimage
import scipy.ndimage.measurements
import random
from test_utils import check_batch, np_type_to_dali


np.random.seed(1234)
random.seed(1234)


def count_outputs(outs):
    if isinstance(outs, dali.pipeline.DataNode):
        return 1
    return len(outs)


data = [
    np.int32([[1, 0, 0, 0], [1, 2, 2, 1], [1, 1, 2, 0], [2, 0, 0, 1]]),
    np.int32([[0, 3, 3, 0], [1, 0, 1, 2], [0, 1, 1, 0], [0, 2, 0, 1], [0, 2, 2, 1]]),
]


def test_num_output():
    """Test that a proper number of outputs is produced, depending on arguments"""
    inp = fn.external_source(data, batch=False)
    assert len(fn.segmentation.random_object_bbox(inp)) == 2
    for label_out_param in [None, False, True]:
        label_out = 1 if label_out_param else 0
        for format, num_box_outputs in [("anchor_shape", 2), ("start_end", 2), ("box", 1)]:
            assert (
                count_outputs(
                    fn.segmentation.random_object_bbox(
                        inp, format=format, output_class=label_out_param
                    )
                )
                == label_out + num_box_outputs
            )


@nottest
def _test_use_foreground(classes, weights, bg):
    inp = fn.external_source(data, batch=False, cycle="quiet")
    pipe = dali.pipeline.Pipeline(10, 4, 0, 12345)
    pipe_outs = fn.segmentation.random_object_bbox(
        inp,
        output_class=True,
        foreground_prob=1,
        classes=classes,
        class_weights=weights,
        background=bg,
    )
    pipe.set_outputs(*pipe_outs)
    outs = pipe.run()
    for i in range(len(outs[2])):
        assert outs[2].at(i) != (bg or 0)


def test_use_foreground():
    """Test that a foreground box is returned when required (prob=1) and possible (fixed data)"""
    for classes, weights, bg in [
        (None, None, None),
        (None, None, 1),
        ([1, 2, 3], None, None),
        (None, [1, 1, 1], None),
        (None, [1, 1, 1], 0),
        ([1, 2, 3], [1, 1, 1], None),
    ]:
        yield _test_use_foreground, classes, weights, bg


def objects2boxes(objects, input_shape):
    if len(objects) == 0:
        return np.int32([[0] * len(input_shape) + list(input_shape)])

    return np.int32([[s.start for s in obj] + [s.stop for s in obj] for obj in objects])


def box_extent(box):
    n = len(box) // 2
    return box[n:] - box[:n]


def box_volume(box):
    return np.prod(box_extent(box))


def box_in_k_largest(boxes, box, k):
    """Returns True if `box` is one of `k` largest boxes in `boxes`. If there are ties that
    extend beyond k, they are included."""
    if len(boxes) == 0:
        return False
    boxes = sorted(boxes, reverse=True, key=box_volume)
    n = len(boxes)
    prev = box_volume(boxes[0])
    for i in range(n):
        vol = box_volume(boxes[i])
        if i >= k:
            if vol < prev:
                break
        prev = vol
        if np.array_equal(boxes[i], box):
            return True
    return False


def all_boxes(array, classes=None, background=None):
    if classes is not None:
        labels = classes
        assert background not in labels
    else:
        if background is None:
            background = 0
        labels = list(np.unique(array))
        try:
            labels.remove(background)
        except ValueError:
            pass

    objects = []
    for lbl in labels:
        mask = array == lbl
        cc, _ = scipy.ndimage.measurements.label(mask)
        objs = scipy.ndimage.find_objects(cc)
        if len(objs) > 0 and objs[0] is not None:
            objects += objs
    return objects2boxes(objects, array.shape)


def class_boxes(array, label):
    mask = array == label
    cc, _ = scipy.ndimage.measurements.label(mask)
    objects = scipy.ndimage.find_objects(cc)
    return objects2boxes(objects, array.shape)


def axis_indices(shape, axis):
    r = np.arange(shape[axis])
    r = np.expand_dims(r, list(range(0, axis)) + list(range(axis + 1, len(shape))))
    rep = list(shape)
    rep[axis] = 1
    r = np.tile(r, rep)
    return r


def indices(shape):
    return np.stack([axis_indices(shape, axis) for axis in range(len(shape))], len(shape))


def generate_data(shape, num_classes, blobs_per_class):
    """Generates blobs_per_class gaussian blobs in ND `shape`-shaped array.
    Each point is assigned a class at which the maximum blob intensity occurred - or background,
    if intensity is below certain threshold. The threshold is adjusted to maintain a preset
    percentage of background"""

    radii = np.array([shape])
    mean = np.random.random([num_classes, blobs_per_class, len(shape)]) * radii
    sigma = (np.random.random([num_classes, blobs_per_class, len(shape)]) * 0.8 + 0.2) * radii / 2

    mean = np.expand_dims(mean, list(range(2, len(shape) + 2)))
    isigma = 1 / np.expand_dims(sigma, list(range(2, len(shape) + 2)))

    pos = np.expand_dims(indices(shape), [0, 1])

    g = np.exp(-np.sum(((pos - mean) * isigma) ** 2, axis=-1))
    g = np.max(g, axis=1)  # sum over blobs within class

    maxfg = np.max(g, axis=0)

    min_bg = 0.5
    max_bg = 0.7
    bg_lo = 0
    bg_hi = 1
    volume = np.prod(shape)
    while bg_hi - bg_lo > 1e-2:
        threshold = (bg_lo + bg_hi) / 2
        bg_fraction = np.count_nonzero(maxfg < threshold) / volume
        if bg_fraction < min_bg:
            bg_lo = threshold
        elif bg_fraction > max_bg:
            bg_hi = threshold
        else:
            break

    label = np.argmax(g, axis=0) + 1
    label[maxfg < threshold] = 0

    return label


def generate_samples(num_samples, ndim, dtype):
    samples = []
    for i in range(num_samples):
        shape = list(np.random.randint(5, 13, [ndim]))
        num_classes = np.random.randint(1, 10)
        blobs_per_class = np.random.randint(1, 10)
        samples.append(generate_data(shape, num_classes, blobs_per_class).astype(dtype))
    return samples


def batch_generator(batch_size, ndim, dtype):
    """Returns a generator that generates completely new data each time it's called"""

    def gen():
        # batch_size = np.random.randint(1, max_batch_size+1)
        return generate_samples(batch_size, ndim, dtype)

    return gen


def sampled_dataset(dataset_size, batch_size, ndim, dtype):
    """Returns a generator that returns random samples from a pre-generated dataset"""
    data = generate_samples(dataset_size, ndim, dtype)

    def gen():
        # batch_size = np.random.randint(1, max_batch_size+1)
        return [random.choice(data) for _ in range(batch_size)]

    return gen


def random_background():
    return fn.random.uniform(range=(-5, 10), dtype=dali.types.INT32, seed=12321)


def random_classes(background):
    def get_classes():
        tmp = list(np.flatnonzero(np.random.random([10]) > 0.5))
        try:
            tmp.remove(background)
        except ValueError:
            pass  # Python, Y U no have try_remove?
        return np.int32(tmp)

    return fn.external_source(get_classes, batch=False)


def random_weights():
    def get_weights():
        tmp = np.random.random(np.random.randint(1, 10)).astype(np.float32)
        return tmp

    return fn.external_source(get_weights, batch=False)


def random_threshold(ndim):
    return fn.random.uniform(range=(1, 5), shape=[ndim], dtype=dali.types.INT32, seed=13231)


def contains_box(boxes, box):
    return (boxes == box).all(axis=1).any()


def convert_boxes(outs, format):
    if format == "box":
        return outs[0]
    elif format == "start_end":
        return [np.concatenate([start, end]) for start, end in zip(outs[0], outs[1])]
    elif format == "anchor_shape":
        return [np.concatenate([anchor, anchor + shape]) for anchor, shape in zip(outs[0], outs[1])]
    else:
        raise ValueError("Test error - unexpected format: {}".format(format))


@nottest
def _test_random_object_bbox_with_class(
    max_batch_size,
    ndim,
    dtype,
    format=None,
    fg_prob=None,
    classes=None,
    weights=None,
    background=None,
    threshold=None,
    k_largest=None,
    cache=None,
):
    pipe = dali.Pipeline(max_batch_size, 4, device_id=None, seed=4321)
    background_out = 0 if background is None else background
    classes_out = np.int32([]) if classes is None else classes
    weights_out = np.int32([]) if weights is None else weights
    threshold_out = np.int32([]) if threshold is None else threshold

    if cache:
        source = sampled_dataset(2 * max_batch_size, max_batch_size, ndim, dtype)
    else:
        source = batch_generator(max_batch_size, ndim, dtype)

    with pipe:
        inp = fn.external_source(source)
        if isinstance(background, dali.pipeline.DataNode) or (
            background is not None and background >= 0
        ):
            inp = fn.cast(inp + (background_out + 1), dtype=np_type_to_dali(dtype))
        # preconfigure
        op = ops.segmentation.RandomObjectBBox(
            format=format,
            foreground_prob=fg_prob,
            classes=classes,
            class_weights=weights,
            background=background,
            threshold=threshold,
            k_largest=k_largest,
            seed=1234,
        )
        outs1 = op(inp, cache_objects=cache)
        outs2 = op(inp, output_class=True)
        if not isinstance(outs1, list):
            outs1 = [outs1]
        # the second instance should have always at least 2 outputs
        assert isinstance(outs2, (list, tuple))
        outputs = [inp, classes_out, weights_out, background_out, threshold_out, *outs1, *outs2]
        pipe.set_outputs(*outputs)

    format = format or "anchor_shape"

    for _ in range(50):
        inp, classes_out, weights_out, background_out, threshold_out, *outs = pipe.run()
        nout = (len(outs) - 1) // 2
        outs1 = outs[:nout]
        outs2 = outs[nout:]
        for i in range(len(outs1)):
            check_batch(outs1[i], outs2[i])

        # Iterate over indices instead of elements, because normal iteration
        # causes an exception to be thrown in native code, making debugging near impossible.
        outs = tuple([np.array(out[i]) for i in range(len(out))] for out in outs1)
        box_class_labels = [np.int32(outs2[-1][i]) for i in range(len(outs2[-1]))]

        boxes = convert_boxes(outs, format)

        for i in range(len(inp)):
            in_tensor = inp.at(i)
            class_labels = classes_out.at(i)
            if background is not None or classes is None:
                background_label = background_out.at(i)
            else:
                background_label = 0 if 0 not in class_labels else np.min(class_labels) - 1

            label = box_class_labels[i]
            if classes is not None:
                assert label == background_label or label in list(class_labels)

            is_foreground = label != background_label
            cls_boxes = class_boxes(in_tensor, label if is_foreground else None)

            if is_foreground:
                ref_boxes = cls_boxes
                if threshold is not None:
                    extent = box_extent(boxes[i])
                    thr = threshold_out.at(i)
                    assert np.all(extent >= thr)
                    ref_boxes = list(filter(lambda box: np.all(box_extent(box) >= thr), cls_boxes))
                if k_largest is not None:
                    assert box_in_k_largest(ref_boxes, boxes[i], k_largest)
            assert contains_box(cls_boxes, boxes[i])


def test_random_object_bbox_with_class():
    np.random.seed(12345)
    types = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32]

    formats = [None, "anchor_shape", "start_end", "box"]
    fmt = 0
    for bg in [None, 0, -1, 5, random_background()]:
        if bg is None or isinstance(bg, int):
            class_opt = [None, [0], [1], [2, 4, 5, 7]]
            for x in class_opt:
                if isinstance(x, list) and bg in x:
                    x.remove(bg)
            if [] in class_opt:
                class_opt.remove([])
            # putting this in the list interfered with remove
            class_opt.append(random_classes(0 if bg is None else bg))
        else:
            class_opt = [None]
        for classes in class_opt:
            if classes is None:
                weights_opt = [None, [1], [0.5, 1, 0.1, 0.2], random_weights()]
            elif isinstance(classes, list):
                weights_opt = [None, list(range(1, 1 + len(classes)))]
            else:
                weights_opt = [None]

            for weights in weights_opt:
                ndim = np.random.randint(1, 5)

                threshold_opt = [None, 3, list(range(1, 1 + ndim)), random_threshold(ndim)]
                threshold = random.choice(threshold_opt)
                k_largest_opt = [None, 1, 2, 5]
                k_largest = random.choice(k_largest_opt)

                fg_prob_opt = [None, 0.1, 0.7, fn.random.uniform(range=(0, 1), seed=1515)]
                fg_prob = random.choice(fg_prob_opt)

                format = formats[fmt]
                fmt = (fmt + 1) % len(formats)
                dtype = random.choice(types)
                cache = np.random.randint(2) == 1
                yield (
                    _test_random_object_bbox_with_class,
                    4,
                    ndim,
                    dtype,
                    format,
                    fg_prob,
                    classes,
                    weights,
                    bg,
                    threshold,
                    k_largest,
                    cache,
                )


@nottest
def _test_random_object_bbox_ignore_class(
    max_batch_size, ndim, dtype, format=None, background=None, threshold=None, k_largest=None
):
    pipe = dali.Pipeline(max_batch_size, 4, device_id=None, seed=4321)
    background_out = 0 if background is None else background
    threshold_out = np.int32([]) if threshold is None else threshold

    with pipe:
        inp = fn.external_source(batch_generator(max_batch_size, ndim, dtype))
        outs = fn.segmentation.random_object_bbox(
            inp,
            format=format,
            ignore_class=True,
            background=background,
            seed=1234,
            threshold=threshold,
            k_largest=k_largest,
        )
        if not isinstance(outs, list):
            outs = [outs]
        pipe.set_outputs(inp, background_out, threshold_out, *outs)

    format = format or "anchor_shape"

    for _ in range(50):
        inp, background_out, threshold_out, *outs = pipe.run()

        # Iterate over indices instead of elements, because normal iteration
        # causes an exception to be thrown in native code, making debugging near impossible.
        outs = tuple([np.array(out[i]) for i in range(len(out))] for out in outs)

        boxes = convert_boxes(outs, format)

        for i in range(len(inp)):
            in_tensor = inp.at(i)
            background_label = background_out.at(i)

            ref_boxes = all_boxes(in_tensor, None, background_label)
            if threshold is not None:
                thr = threshold_out.at(i)
                ref_boxes = list(filter(lambda box: np.all(box_extent(box) >= thr), ref_boxes))
                if len(ref_boxes) == 0:
                    ref_boxes = np.int32([[0] * len(in_tensor.shape) + list(in_tensor.shape)])
            if k_largest is not None:
                assert box_in_k_largest(ref_boxes, boxes[i], k_largest)
            else:
                assert contains_box(ref_boxes, boxes[i])


def test_random_object_bbox_ignore_class():
    np.random.seed(43210)
    types = [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32]
    for bg in [None, 0, -1, 5, random_background()]:
        ndim = np.random.randint(1, 5)
        dtype = random.choice(types)
        for format in [None, "anchor_shape", "start_end", "box"]:
            threshold_opt = [None, 3, list(range(1, 1 + ndim)), random_threshold(ndim)]
            threshold = random.choice(threshold_opt)
            k_largest_opt = [None, 1, 2, 5]
            k_largest = random.choice(k_largest_opt)

            yield (
                _test_random_object_bbox_ignore_class,
                5,
                ndim,
                dtype,
                format,
                bg,
                threshold,
                k_largest,
            )


@nottest
def _test_random_object_bbox_auto_bg(fg_labels, expected_bg):
    """Checks that a correct background labels is chosen:
    0, if 0 is not present in the list of foreground classes
    smallest label - 1 if 0 is present
    if the smallest label -1 overflows, decrement the label until no collision
    """
    pipe = dali.Pipeline(batch_size=1, num_threads=1, device_id=0, seed=1234)
    data = np.uint32([0, 1, 2, 3])

    box, label = fn.segmentation.random_object_bbox(
        data, foreground_prob=1e-9, format="box", output_class=1, classes=fg_labels
    )

    pipe.set_outputs(box, label)
    _, labels = pipe.run()
    assert int(labels.at(0)) == expected_bg


def test_random_object_bbox_auto_bg():
    for fg, expected_bg in [
        ([1, 2, 3], 0),
        ([0, 1, 2], -1),
        ([-1, 1], 0),
        ([0, -5], -6),
        ([-0x80000000, 0x7FFFFFFF], 0),
        ([-0x80000000, 0x7FFFFFFF, 0, 0x7FFFFFFE], 0x7FFFFFFD),
    ]:
        yield _test_random_object_bbox_auto_bg, fg, expected_bg


@nottest
def _test_err_args(**kwargs):
    pipe = dali.Pipeline(batch_size=1, num_threads=1, device_id=0, seed=1234)
    inp = fn.external_source(data, batch=False)
    outs = fn.segmentation.random_object_bbox(inp, **kwargs)
    pipe.set_outputs(*outs)
    pipe.run()


def test_err_classes_bg():
    with assert_raises(RuntimeError, glob="Class label 0 coincides with background label"):
        _test_err_args(classes=[0, 1, 2, 3], background=0)


def test_err_classes_weights_length_clash():
    error_msg = (
        r"If both ``classes`` and ``class_weights`` are provided, their shapes must "
        r"match. Got:\s+classes.shape = \{4\}\s+weights.shape = \{3\}"
    )
    with assert_raises(RuntimeError, regex=error_msg):
        _test_err_args(classes=[0, 1, 2, 3], class_weights=np.float32([1, 2, 3]))
    with assert_raises(RuntimeError, regex=error_msg):
        _test_err_args(classes=np.int32([0, 1, 2, 3]), class_weights=[3, 2, 1])


def test_err_classes_ignored():
    with assert_raises(
        RuntimeError, glob="Class-related arguments * cannot be used when ``ignore_class`` is True"
    ):
        _test_err_args(classes=[0, 1, 2, 3], ignore_class=True)


def test_err_k_largest_nonpositive():
    with assert_raises(RuntimeError, glob="``k_largest`` must be at least 1; got -1"):
        _test_err_args(k_largest=-1)
    with assert_raises(RuntimeError, glob="``k_largest`` must be at least 1; got 0"):
        _test_err_args(k_largest=0)


def test_err_threshold_dim_clash():
    with assert_raises(
        RuntimeError,
        glob='Argument "threshold" expected shape 2 but got 5 values, '
        "which can't be interpreted as the expected shape.",
    ):
        _test_err_args(threshold=[1, 2, 3, 4, 5])


@attr("slow")
def slow_test_large_data():
    yield _test_random_object_bbox_with_class, 4, 5, np.int32, None, 1.0, [
        1,
        2,
        3,
    ], None, None, None, 10
