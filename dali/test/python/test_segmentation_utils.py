# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
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

import numpy as np
import random

def make_batch_select_masks(batch_size, num_masks_range = (1, 10), coords_per_mask_range = (3, 40), coord_ndim = 2, coord_dtype = np.float32):
    masks_meta = []
    masks_coords = []
    selected_masks = []
    for i in range(batch_size):
        nmasks = random.randint(*num_masks_range)
        available_masks = list(range(nmasks))
        selected_masks.append(np.array(random.sample(available_masks, random.randint(1, nmasks)), dtype = np.int32))
        coord_count = 0
        mask_idx = 0
        curr_masks_meta = np.zeros([nmasks, 3], dtype=np.int32)
        for m in range(nmasks):
            ncoords = random.randint(*coords_per_mask_range)
            curr_masks_meta[m, :] = (mask_idx, coord_count, coord_count + ncoords - 1)
            coord_count = coord_count + ncoords
            mask_idx = mask_idx + 1
        masks_meta.append(curr_masks_meta)
        masks_coords.append(np.array(np.random.rand(coord_count, coord_ndim), dtype=coord_dtype))
    return masks_meta, masks_coords, selected_masks

