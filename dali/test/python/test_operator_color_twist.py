from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy as np
import os

from test_utils import compare_pipelines
from test_utils import RandomDataIterator
from test_utils import RandomDataIterator

class ColorTwistPipeline(Pipeline):
    def __init__(self, batch_size, seed, data_iterator, old=False, num_threads=1, device_id=0):
        super(ColorTwistPipeline, self).__init__(batch_size, num_threads, device_id, seed=seed)
        self.iterator = data_iterator
        self.input = ops.ExternalSource()
        self.hue = ops.Uniform(range=[-20., 20.])
        self.sat = ops.Uniform(range=[0., 1.])
        self.bri = ops.Uniform(range=[0., 2.])
        self.con = ops.Uniform(range=[0., 2.])
        self.color_twist = ops.OldColorTwist(device="gpu") if old else ops.ColorTwist(device="gpu") 


    def define_graph(self):
        self.images = self.input()
        return self.color_twist(self.images.gpu(), hue=self.hue(), saturation=self.sat(), brightness=self.bri(), contrast=self.con())
    
    def iter_setup(self):
        self.feed_input(self.images, self.iterator.next(), layout="HWC")


def test_color_twist_vs_old():
    batch_size = 32
    seed = 2139
    rand_it1 = RandomDataIterator(batch_size, shape=(1024, 512, 3))
    rand_it2 = RandomDataIterator(batch_size, shape=(1024, 512, 3))
    compare_pipelines(ColorTwistPipeline(batch_size, seed, iter(rand_it1)),
                      ColorTwistPipeline(batch_size, seed, iter(rand_it2), old=True),
                      batch_size=batch_size, N_iterations=64, eps=1)

