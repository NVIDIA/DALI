

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types

import numpy as np

sample_size = 5
premade_batch = [np.array(x) for x in range(sample_size)]

class OneHotPipeline(Pipeline):
    def __init__(self, no_classes, num_threads=1):
        super(OneHotPipeline, self).__init__(sample_size,
                                            num_threads,
                                            0)
        self.ext_src = ops.ExternalSource() 
        self.one_hot = ops.OneHot(nclasses=no_classes, device="cpu")


    def define_graph(self):
        self.data = self.ext_src()
        return self.one_hot(self.data)

    def iter_setup(self):
        self.feed_input(self.data, premade_batch);


def one_hot(input):
    outp = np.zeros([sample_size, sample_size], dtype=int)
    for i in range(sample_size):
        outp[i,input[i]] = 1
    return outp

def check_one_hot_operator():
    pipeline = OneHotPipeline(no_classes=sample_size)
    pipeline.build()
    outputs = pipeline.run()
    reference = one_hot(premade_batch)
    outputs = outputs[0]
    for i in range(sample_size):
        out_array = outputs.at(i)
        for j in range(sample_size):
            # compare onehot encoding with python implementation
            assert(out_array[j] == reference[i,j])

    

if __name__ == "__main__":
    check_one_hot_operator()
