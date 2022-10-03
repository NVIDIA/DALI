# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
import os
import numpy as np
import time

def measure(decoder_f, backend, dir, batch, iters=10):
    @pipeline_def
    def pipe():
        encoded, _ = fn.readers.file(file_root=dir)
        images = decoder_f(encoded, device=backend, output_type=types.DALIImageType.RGB)
        return images
    p = pipe(batch_size=batch, num_threads=16, device_id=0)
    p.build()

    for _ in range(3):
        # warmup
        out = p.run()
    if backend == 'mixed': out[0][0].as_cpu()  # trying to force a sync

    t0 = time.time()
    for _ in range(iters):
        out = p.run()
    if backend == 'mixed': out[0][0].as_cpu()  # trying to force a sync
    t1 = time.time()
    del p

    nimages = iters * batch
    fps = nimages/(t1 - t0)
    return fps

def get_data_path(format):
    return os.path.join(os.environ['DALI_EXTRA_PATH'], 'db/single', format)

for backend in ('mixed',):
    for batch, iters in ((256, 10), (16, 100)):
        results = []
        formats = ('jpeg', 'jpeg2k', 'tiff', 'png', 'bmp', 'webp', 'pnm', 'mixed')
        for format in formats:
            old = measure(fn.decoders.image, backend, get_data_path(format), batch, iters=iters)
            new = measure(fn.experimental.decoders.image, backend, get_data_path(format), batch, iters=iters)
            results.append((format, old, new))
            print(backend, format, ':', old, '->', new, 'fps')

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        xs = np.arange(len(formats))
        width = 0.35
        old = ax.bar(xs - width/2, [100 for _ in results], width, label='Old decoder')
        new = ax.bar(xs + width/2, [100 * r[2] / r[1] for r in results], width, label='Imgcodec')
        ax.set_ylabel('FPS')
        ax.set_title(f'Imgcodec vs old decoder performance ({backend}, batch={batch})')
        ax.set_xticks(xs, formats)
        ax.set_yticklabels([])
        ax.legend()
        ax.bar_label(old, labels=[int(r[1]) for r in results], padding=3)
        ax.bar_label(new, labels=[int(r[2]) for r in results], padding=3)
        fig.tight_layout()
        fig.show()
