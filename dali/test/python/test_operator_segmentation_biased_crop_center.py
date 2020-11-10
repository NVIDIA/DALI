import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from test_utils import check_batch, dali_type
import random
from segmentation_test_utils import make_batch_select_masks
from nose.tools import assert_raises

np.random.seed(4321)

def check_biased_crop_center(ndim=2, batch_size=3,
                             min_extent=20, max_extent=50):
    pipe = dali.pipeline.Pipeline(batch_size=batch_size, num_threads=4, device_id=0, seed=1234)
    with pipe:
        # Input mask
        in_shape_dims = [fn.cast(fn.uniform(range=(min_extent, max_extent + 1), shape=(1,), device='cpu'),
                                 dtype=types.INT32) for d in range(ndim)]
        in_shape = fn.cat(*in_shape_dims, axis=0)
        in_mask = fn.cast(fn.uniform(range=(0, 1), device='cpu', shape=in_shape), dtype=types.INT32)

        # Crop dims
        crop_shape = in_shape - 2  # We want to force the center adjustment, therefore the large crop shape

        # Crop centers
        always_fg_center = fn.segmentation.biased_crop_center(in_mask, nonzero=1)
        random_center = fn.segmentation.biased_crop_center(in_mask, nonzero=0)
        coin_flip = fn.coin_flip(probability=0.7)
        biased_center = fn.segmentation.biased_crop_center(in_mask, nonzero=coin_flip)
        always_fg_center_w_sh = fn.segmentation.biased_crop_center(in_mask, nonzero=1, shape=crop_shape)
        random_center_w_sh = fn.segmentation.biased_crop_center(in_mask, nonzero=0, shape=crop_shape)

        # Transforming center to anchor
        anchor = always_fg_center_w_sh - crop_shape // 2
        crop_shape = fn.cast(crop_shape, dtype=types.INT64)  # anchor and shape type should match
        out_mask = fn.slice(in_mask, anchor, crop_shape, axes=tuple(range(ndim)))
    pipe.set_outputs(in_mask, always_fg_center, random_center, coin_flip, biased_center,
                     crop_shape, always_fg_center_w_sh, random_center_w_sh, out_mask)
    pipe.build()
    for iter in range(3):
        outputs = pipe.run()
        for idx in range(batch_size):
            in_mask = outputs[0].at(idx)
            always_fg_center = outputs[1].at(idx).tolist()
            random_center = outputs[2].at(idx).tolist()
            coin_flip = outputs[3].at(idx)
            biased_center = outputs[4].at(idx).tolist()
            crop_shape = tuple(outputs[5].at(idx).tolist())
            always_fg_center_w_sh = outputs[6].at(idx).tolist()
            random_center_w_sh = outputs[7].at(idx).tolist()
            out_mask = outputs[8].at(idx)

            for d in range(ndim):
                assert random_center[d] >= 0 and random_center[d] < in_mask.shape[d]
                assert always_fg_center[d] >= 0 and always_fg_center[d] < in_mask.shape[d]
                assert biased_center[d] >= 0 and biased_center[d] < in_mask.shape[d]

            assert in_mask[tuple(always_fg_center)] > 0
            assert in_mask[tuple(biased_center)] > 0 or not coin_flip

            for d in range(ndim):
                always_fg_center_anchor_d = always_fg_center_w_sh[d] - crop_shape[d] // 2
                random_center_anchor_d = random_center_w_sh[d] - crop_shape[d] // 2
                assert always_fg_center_anchor_d >= 0 and always_fg_center_anchor_d + crop_shape[d] <= in_mask.shape[d]
                assert random_center_anchor_d >= 0 and random_center_anchor_d + crop_shape[d] <= in_mask.shape[d]
            assert out_mask.shape == crop_shape

def test_biased_crop_center():
    for ndim in (2, 3):
        yield check_biased_crop_center, ndim
