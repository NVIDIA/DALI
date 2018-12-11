#!/bin/env python

import numpy as np
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

try:
    from matplotlib import pyplot as plt
    has_matplotlib = True
except ImportError:
    has_matplotlib = False


BATCH_SIZE=4
COUNT=5

VIDEO_FILES=["prepared.mp4"]
ITER=10

class VideoPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data):
        super(VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.input = ops.VideoReader(device="gpu", filenames=data, count=COUNT,
                                     shard_id=0, num_shards=1, random_shuffle=False)


    def define_graph(self):
        output = self.input(name="Reader")
        return output

if __name__ == "__main__":
    pipe = VideoPipe(batch_size=BATCH_SIZE, num_threads=2, device_id=0, data=VIDEO_FILES)
    pipe.build()
    for i in range(ITER):
        print("Iteration " + str(i))
        pipe_out = pipe.run()
        sequences_out = pipe_out[0].asCPU().as_array()
        print(sequences_out.shape)
        print("Got sequence " + str(i*COUNT) + " " + str((i + 1)*COUNT - 1))
        for b in range(BATCH_SIZE):
            batch_sequences = sequences_out[b]
            print(batch_sequences.shape)
            for c in range(COUNT):
                sample_frame = batch_sequences[c]

    frame_to_show = sequences_out[0][0]
    if has_matplotlib:
        plt.imshow(frame_to_show.astype('uint8'), interpolation='bicubic')
        plt.show()
        plt.savefig('saved_frame.png')