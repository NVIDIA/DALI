import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import numpy as np
import scipy.ndimage
import scipy.ndimage.measurements
from test_utils import check_batch
from test_utils import np_type_to_dali
from nose.tools import raises, assert_raises, nottest

np.random.seed(1234)

def count_outputs(outs):
    if isinstance(outs, dali.pipeline.DataNode):
        return 1
    return len(outs)

def test_num_output():
    """Test that a proper number of outputs is produced, depending on arguments"""
    inp = fn.external_source(data, batch=False)
    assert len(fn.segmentation.random_object_bbox(inp)) == 2
    for label_out_param in [None, False, True]:
        label_out = 1 if label_out_param else 0
        assert count_outputs(fn.segmentation.random_object_bbox(inp, format="anchor_shape", output_class=label_out_param)) == 2 + label_out
        assert count_outputs(fn.segmentation.random_object_bbox(inp, format="start_end", output_class=label_out_param)) == 2 + label_out
        assert count_outputs(fn.segmentation.random_object_bbox(inp, format="box", output_class=label_out_param)) == 1 + label_out

def objects2boxes(objects, input_shape):
    if len(objects) == 0:
        return np.int32([[0] * len(input_shape) + list(input_shape)])

    return np.int32([
        [s.start for s in obj] + [s.stop for s in obj] for obj in objects])

def box_extent(box):
    n = len(box) // 2
    return box[n:]-box[:n]

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


def all_boxes(array, classes = None, background = None):
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
    for l in labels:
        mask = array == l
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

data = [
    np.int32([[1, 0, 0, 0],
              [1, 2, 2, 1],
              [1, 1, 2, 0],
              [2, 0, 0, 1]]),

    np.int32([[0, 3, 3, 0],
              [1, 0, 1, 2],
              [0, 1, 1, 0],
              [0, 2, 0, 1],
              [0, 2, 2, 1]])
]


def axis_indices(shape, axis):
    r = np.arange(shape[axis])
    r = np.expand_dims(r, list(range(0, axis)) + list(range(axis+1, len(shape))))
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
    precentage of background"""

    radii = np.array([shape])
    mean = np.random.random([num_classes, blobs_per_class, len(shape)]) * radii
    sigma = (np.random.random([num_classes, blobs_per_class, len(shape)])*0.8 + 0.2) * radii / 2

    mean = np.expand_dims(mean, list(range(2, len(shape) + 2)))
    isigma = 1 / np.expand_dims(sigma, list(range(2, len(shape) + 2)))

    pos = np.expand_dims(indices(shape), [0, 1])

    g = np.exp(-np.sum(((pos - mean) * isigma) ** 2, axis=-1))
    g = np.max(g, axis=1)  # sum over blobs within class

    maxfg = np.max(g, axis = 0)

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

    label = np.argmax(g, axis = 0) + 1
    label[maxfg < threshold] = 0

    return label

def batch_generator(max_batch_size, ndim, dtype):
    def gen():
        batch_size = np.random.randint(1, max_batch_size+1)
        batch = []
        for i in range(batch_size):
            shape = list(np.random.randint(5, 11, [ndim]))
            num_classes = np.random.randint(1, 10)
            blobs_per_class = np.random.randint(1, 10);
            batch.append(generate_data(shape, num_classes, blobs_per_class).astype(dtype))
        return batch
    return gen

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

def contains_box(boxes, box):
    return (boxes == box).all(axis = 1).any()

def convert_boxes(outs, format):
    if format == "box":
        return outs[0]
    elif format == "start_end":
        return [np.concatenate([start, end]) for start, end in zip(outs[0], outs[1])]
    elif format == "anchor_shape":
        return [np.concatenate([anchor, anchor+shape]) for anchor, shape in zip(outs[0], outs[1])]
    else:
        raise ValueError("Test error - unexpected format: {}".format(format))

@nottest
def _test_random_object_bbox_with_class(max_batch_size, ndim, dtype, format=None, fg_prob=None,
                                        classes=None, weights=None, background=None,
                                        threshold=None, k_largest=None):
    pipe = dali.pipeline.Pipeline(max_batch_size, 4, device_id = None, seed=4321)
    background_out = 0 if background is None else background
    classes_out = np.int32([]) if classes is None else classes
    weights_out = np.int32([]) if weights is None else weights
    threshold_out = np.int32([]) if threshold is None else threshold

    with pipe:
        inp = fn.external_source(batch_generator(max_batch_size, ndim, dtype))
        if isinstance(background, dali.pipeline.DataNode) or (background is not None and background >= 0):
            inp = fn.cast(inp + (background_out + 1), dtype=np_type_to_dali(dtype))
        # preconfigure
        op = ops.segmentation.RandomObjectBBox(format=format,
                                               foreground_prob=fg_prob,
                                               classes=classes, class_weights=weights, background=background,
                                               threshold=threshold, k_largest=k_largest,
                                               seed=1234)
        outs1 = op(inp)
        outs2 = op(inp, output_class=True)
        if not isinstance(outs1, list):
            outs1 = [outs1]
        # the second instance should have always at least 2 outputs
        assert isinstance(outs2, (list, tuple))
        outputs = [inp, classes_out, weights_out, background_out, threshold_out, *outs1, *outs2]
        pipe.set_outputs(*outputs)
    pipe.build()

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
                background_label = 0 if 0 not in class_labels else np.min(class_labels)-1

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
    for bg in [None, 0, -1, 5, fn.random.uniform(range=(-5, 10), dtype=dali.types.INT32, seed=12321)]:
        if bg is None or isinstance(bg, int):
            class_opt = [None, [0], [1], [2,4,5,7]]
            for x in class_opt:
                if isinstance(x, list) and bg in x:
                    x.remove(bg)
            if [] in class_opt: class_opt.remove([])
            # putting this in the list interefered with remove
            class_opt.append(random_classes(0 if bg is None else bg))
        else:
            class_opt = [None]
        for classes in class_opt:
            if classes is None:
                weights_opt = [None, [1], [0.5, 1, 0.1, 0.2], random_weights()]
            elif isinstance(classes, list):
                weights_opt = [None, list(range(1, 1+len(classes)))]
            else:
                weights_opt = [None]

            for weights in weights_opt:
                ndim = np.random.randint(1, 5)

                threshold_opt = [None, 3, list(range(1, 1+ndim)), fn.random.uniform(range=(1,5), shape=[ndim], dtype=dali.types.INT32, seed=13231)]
                threshold = threshold_opt[np.random.randint(len(threshold_opt))]
                k_largest_opt = [None, 1, 2, 5]
                k_largest = k_largest_opt[np.random.randint(len(k_largest_opt))]

                fg_prob_opt = [None, 0.1, 0.7, fn.random.uniform(range=(0,1), seed=1515)]
                fg_prob = fg_prob_opt[np.random.randint(len(fg_prob_opt))]

                format = formats[fmt]
                fmt = (fmt + 1) % len(formats)
                dtype = types[np.random.randint(0, len(types))]
                yield _test_random_object_bbox_with_class, 5, ndim, dtype, format, fg_prob, classes, weights, bg, threshold, k_largest

@nottest
def _test_random_object_bbox_ignore_class(max_batch_size, ndim, dtype, format=None, background=None, threshold=None, k_largest=None):
    pipe = dali.pipeline.Pipeline(max_batch_size, 4, device_id = None, seed=4321)
    background_out = 0 if background is None else background
    threshold_out = np.int32([]) if threshold is None else threshold

    with pipe:
        inp = fn.external_source(batch_generator(max_batch_size, ndim, dtype))
        outs = fn.segmentation.random_object_bbox(inp, format=format, ignore_class=True, background=background, seed=1234,
                                                  threshold=threshold, k_largest=k_largest)
        if not isinstance(outs, list):
            outs = [outs]
        pipe.set_outputs(inp, background_out, threshold_out, *outs)
    pipe.build()

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
    for bg in [None, 0, -1, 5, fn.random.uniform(range=(-5, 10), dtype=dali.types.INT32, seed=1313)]:
        ndim = np.random.randint(1, 5)
        dtype = types[np.random.randint(0, len(types))]
        for format in [None, "anchor_shape", "start_end", "box"]:
            threshold_opt = [None, 3, list(range(1, 1+ndim)), fn.random.uniform(range=(1,5), shape=[ndim], dtype=dali.types.INT32, seed=3214)]
            threshold = threshold_opt[np.random.randint(len(threshold_opt))]
            k_largest_opt = [None, 1, 2, 5]
            k_largest = k_largest_opt[np.random.randint(len(k_largest_opt))]

            yield _test_random_object_bbox_ignore_class, 5, ndim, dtype, format, bg, threshold, k_largest
