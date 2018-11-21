#!/bin/env python

import numpy as np
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
from matplotlib import pyplot as plt


BATCH_SIZE=2
COUNT=4
HEIGHT=720
WIDTH=1280

VIDEO_FILES=["prepared.mp4"]
ITER=142

class VideoPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, data):
        super(VideoPipe, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.input = ops.VideoReader(device="gpu", filenames=data,
                                     count=COUNT, height=HEIGHT, width=WIDTH,
                                     shard_id=0, num_shards=1, random_shuffle=False)


    def define_graph(self):
        output = self.input(name="Reader")
        return output

if __name__ == "__main__":
    pipe = VideoPipe(batch_size=BATCH_SIZE, num_threads=2, device_id=0, data=VIDEO_FILES)
    pipe.build()
    for i in range(ITER):
        pipe_out = pipe.run()
        sequences_out = pipe_out[0].asCPU().at(0)
        print("Got sequence " + str(i*COUNT) + " " + str((i + 1)*COUNT - 1))
        for b in range(BATCH_SIZE):
            batch_sequences = sequences_out[b]
            print(batch_sequences.shape)
            for c in range(COUNT):
                sample_frame = batch_sequences[c]
                print('Sum of all elements for b ' + str(b) + ' frame ' + str(c) + ' = ' + str(np.sum(sample_frame)))

    frame_to_show = sequences_out[0][0]
    plt.imshow(frame_to_show.astype('uint8'), interpolation='bicubic')
    plt.show()
    plt.savefig('saved_frame.png')