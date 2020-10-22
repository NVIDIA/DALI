import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn
from test_utils import check_batch, dali_type
import random
from test_segmentation_utils import make_batch_select_masks
from nose.tools import assert_raises

random.seed(1234)
np.random.seed(4321)

def check_select_masks(batch_size, num_masks_range = (1, 10), coords_per_mask_range = (3, 40), coord_ndim = 2, coord_dtype = np.float32, reindex_masks = False):
    def get_data_source(*args, **kwargs):
        return lambda: make_batch_select_masks(*args, **kwargs)
    pipe = dali.pipeline.Pipeline(batch_size=batch_size, num_threads=4, device_id=0, seed=1234)
    with pipe:
        masks_meta, masks_coords, selected_masks = fn.external_source(
            source = get_data_source(batch_size, num_masks_range=num_masks_range,
            coords_per_mask_range=coords_per_mask_range, coord_ndim=coord_ndim, coord_dtype=coord_dtype),
            num_outputs = 3, device='cpu'
        )
        out_masks_meta, out_masks_coords = fn.segmentation.select_masks(
            selected_masks, masks_meta, masks_coords, reindex_masks=reindex_masks
        )
    pipe.set_outputs(masks_meta, masks_coords, selected_masks, out_masks_meta, out_masks_coords)
    pipe.build()
    for iter in range(3):
        outputs = pipe.run()
        for idx in range(batch_size):
            in_masks_meta = outputs[0].at(idx)
            in_masks_coords = outputs[1].at(idx)
            selected_masks = outputs[2].at(idx)
            out_masks_meta = outputs[3].at(idx)
            out_masks_coords = outputs[4].at(idx)

            in_masks_meta_dict = {}
            for k in range(in_masks_meta.shape[0]):
                mask_id = in_masks_meta[k, 0]
                in_masks_meta_dict[mask_id] = (in_masks_meta[k, 1], in_masks_meta[k, 2])

            if reindex_masks:
                index_map = {}
                for idx in range(len(selected_masks)):
                    index_map[selected_masks[idx]] = idx

            coord_count = 0
            for m in range(len(selected_masks)):
                mask_id = selected_masks[m]
                in_coord_start, in_coord_end = in_masks_meta_dict[mask_id]
                in_ncoords = in_coord_end + 1 - in_coord_start

                expected_out_mask_id = index_map[mask_id] if reindex_masks else mask_id
                out_mask_id, out_coord_start, out_coord_end = out_masks_meta[m]
                assert out_mask_id == expected_out_mask_id
                assert out_coord_start == coord_count
                assert out_coord_end == (coord_count + in_ncoords - 1)
                coord_count = coord_count + in_ncoords

                expected_out_coords = in_masks_coords[in_coord_start:in_coord_end+1]
                out_coords = out_masks_coords[out_coord_start:out_coord_end+1]
                assert (expected_out_coords == out_coords).all()

def test_select_masks():
    num_masks_range = (1, 10)
    coords_per_mask_range = (3, 40)
    for batch_size in [1, 3]:
        for coord_ndim in [2, 3, 6]:
            for coord_dtype in [np.float, random.choice([np.int8, np.int16, np.int32, np.int64])]:
                reindex_masks = random.choice([False, True])
                yield check_select_masks, batch_size, num_masks_range, coords_per_mask_range, \
                    coord_ndim, coord_dtype, reindex_masks

def check_select_masks_wrong_input(data_source_fn, batch_size=1, reindex_masks=False):
    pipe = dali.pipeline.Pipeline(batch_size=batch_size, num_threads=4, device_id=0, seed=1234)
    with pipe:
        masks_meta, masks_coords, selected_masks = fn.external_source(
            source = data_source_fn, num_outputs = 3, device='cpu'
        )
        out_masks_meta, out_masks_coords = fn.segmentation.select_masks(
            selected_masks, masks_meta, masks_coords, reindex_masks=reindex_masks
        )
    pipe.set_outputs(masks_meta, masks_coords, selected_masks, out_masks_meta, out_masks_coords)
    pipe.build()
    with assert_raises(RuntimeError):
        outputs = pipe.run()

def test_select_masks_wrong_mask_ids():
    def test_data():
        masks_meta = [np.array([[0, 0, 2], [1, 3, 5], [2, 6, 8]], dtype=np.int32)]
        masks_coords = [np.array(np.random.rand(9, 2), dtype=np.float32)]
        selected_masks = [np.array([10, 11], dtype = np.int32)]  # out of bounds ids
        return masks_meta, masks_coords, selected_masks
    check_select_masks_wrong_input(lambda: test_data())

def test_select_masks_wrong_mask_meta_dim():
    def test_data():
        # Expects 3 integers, not 4 
        masks_meta = [np.array([[0, 0, 2, -1], [1, 3, 5, -1], [2, 6, 8, -1]], dtype=np.int32)]
        masks_coords = [np.array(np.random.rand(9, 2), dtype=np.float32)]
        selected_masks = [np.array([0], dtype=np.int32)]
        return masks_meta, masks_coords, selected_masks
    fn = lambda: test_data()
    check_select_masks_wrong_input(lambda: test_data())

def test_select_masks_wrong_coord_ids():
    def test_data():
        masks_meta = [np.array([[0, 0, 20]], dtype=np.int32)]  # Out of bounds coordinates
        masks_coords = [np.array(np.random.rand(3, 2), dtype=np.float32)]  # Only 3 coords
        selected_masks = [np.array([0], dtype=np.int32)]
        return masks_meta, masks_coords, selected_masks
    check_select_masks_wrong_input(lambda: test_data())
