import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import pipeline_def
import numpy as np


def f(x):
    return x


@pipeline_def(batch_size=256, num_threads=1, seed=0)
def pipeline():
    x = types.Constant(np.full((1,), 0))
    x = fn.python_function(x, function=f, num_outputs=1)
    y = types.Constant(np.full((1024, 720), 0))

    return x + y


p = None


def test_shutdown():
    global p
    p = pipeline(device_id=None)
    p.build()
    p.run()
