import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import nvidia.dali.math as math
from test_utils import check_batch, dali_type
import random
from nose.tools import assert_raises

np.random.seed(4321)

def random_shape(min_sh, max_sh, ndim):
  return np.array(
      [np.random.randint(min_sh, max_sh) for s in range(ndim)],
      dtype=np.int32
    )

def check_roi_random_crop(ndim=2, batch_size=3,
                          roi_min_start = 0, roi_max_start = 100,
                          roi_min_extent = 20, roi_max_extent = 50,
                          crop_min_extent = 20, crop_max_extent = 50,
                          in_shape_min = 400, in_shape_max = 500,
                          use_in_shape_arg = False, use_shape_like_in = False,
                          niter=3):
    pipe = dali.pipeline.Pipeline(batch_size=batch_size, num_threads=4, device_id=0, seed=1234)
    with pipe:
        in_shape_out = None
        in_shape_arg = None
        inputs = []
        if use_in_shape_arg or use_shape_like_in:
            assert in_shape_min < in_shape_max
            shape_gen_f = lambda: random_shape(in_shape_min, in_shape_max, ndim)
            if use_shape_like_in:
                shape_like_in = dali.fn.external_source(lambda: np.zeros(shape_gen_f()), 
                                                        device='cpu', batch=False)
                in_shape_out = dali.fn.shapes(shape_like_in)
                inputs += [shape_like_in]
            elif use_in_shape_arg:
                in_shape_arg = dali.fn.external_source(shape_gen_f, batch=False)
                in_shape_out = in_shape_arg
        crop_shape = fn.random.uniform(range=(crop_min_extent, crop_max_extent + 1), 
                                       shape=(ndim,), dtype=types.INT32, device='cpu')
        roi_shape = fn.random.uniform(range=(roi_min_extent, roi_max_extent + 1),
                                      shape=(ndim,), dtype=types.INT32, device='cpu')
        roi_start = fn.random.uniform(range=(roi_min_start, roi_max_start + 1),
                                      shape=(ndim,), dtype=types.INT32, device='cpu')
        crop_start = fn.roi_random_crop(*inputs, crop_shape=crop_shape,
                                        roi_start=roi_start, roi_shape=roi_shape,
                                        in_shape=in_shape_arg, device='cpu')
    outputs = [roi_start, roi_shape, crop_start, crop_shape]
    if use_in_shape_arg or use_shape_like_in:
        outputs += [in_shape_out]
    pipe.set_outputs(*outputs)
    pipe.build()
    for _ in range(niter):
        outputs = pipe.run()
        for idx in range(batch_size):
            roi_start = np.array(outputs[0][idx]).tolist()
            roi_shape = np.array(outputs[1][idx]).tolist()
            crop_start = np.array(outputs[2][idx]).tolist()
            crop_shape = np.array(outputs[3][idx]).tolist()

            in_shape = None
            if use_in_shape_arg or use_shape_like_in:
                in_shape = np.array(outputs[4][idx]).tolist()

            roi_end = [roi_start[d] + roi_shape[d] for d in range(ndim)]
            crop_end = [crop_start[d] + crop_shape[d] for d in range(ndim)]

            for d in range(ndim):
                if in_shape is not None:
                    assert crop_start[d] >= 0
                    assert crop_end[d] <= in_shape[d]

                if crop_shape[d] >= roi_shape[d]:
                    assert roi_start[d] >= crop_start[d]
                    assert roi_end[d] <= crop_end[d] 
                else:
                    assert crop_start[d] >= roi_start[d]
                    assert crop_end[d] <= roi_end[d]

def test_random_mask_pixel():
    batch_size = 3
    niter = 3
    for ndim in (2, 3):
        for roi_start_min, roi_start_max, roi_extent_min, roi_extent_max, \
            crop_extent_min, crop_extent_max in \
                [(20, 50, 10, 20, 30, 40),
                 (20, 50, 100, 140, 30, 40),
                 (0, 1, 10, 20, 80, 100)]:
            in_shape_min = 250
            in_shape_max = 300
            yield check_roi_random_crop, ndim, batch_size, roi_start_min, roi_start_max, roi_extent_min, roi_extent_max, \
                crop_extent_min, crop_extent_max, in_shape_min, in_shape_max, True, False, niter
            yield check_roi_random_crop, ndim, batch_size, roi_start_min, roi_start_max, roi_extent_min, roi_extent_max, \
                crop_extent_min, crop_extent_max, in_shape_min, in_shape_max, False, True, niter
            yield check_roi_random_crop, ndim, batch_size, roi_start_min, roi_start_max, roi_extent_min, roi_extent_max, \
                crop_extent_min, crop_extent_max, None, None, False, False, niter
