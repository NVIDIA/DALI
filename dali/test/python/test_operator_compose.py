import nvidia.dali as dali
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import numpy as np

def test_unified_arg_placement():
    batch_size = 30

    pipe = Pipeline(batch_size,1,None)
    with pipe:
        u = ops.Uniform()(range=(1,2), shape=3)
        tr = ops.transforms.Translation(offset=u, device="cpu")
        pipe.set_outputs(tr(), u)
    pipe.build()
    matrices, offsets = pipe.run()
    assert len(matrices) == batch_size
    for i in range(len(matrices)):
        offset = offsets.at(i)
        matrix = matrices.at(i)
        assert offset.shape == (3,)
        for j in range(3):
            assert offset[j] >= 1 and offset[j] < 2  # check that it's not all zeros or sth
        T = offset[:,np.newaxis]  # convert to a columnn
        assert np.array_equal(matrix, np.concatenate([np.identity(3), T], axis=1))

def test_compose():
    batch_size = 3
    pipe = Pipeline(batch_size,1,None)

    u = ops.Uniform()(range=(1,2), shape=3)
    c1 = ops.Compose([
        ops.transforms.Translation(offset=u),
        ops.transforms.Scale(scale=[1,1,-1])
    ])
    c2 = ops.Compose([
        c1,
        ops.transforms.Rotation(angle=90, axis=[0,0,1])
    ])
    pipe.set_outputs(c2(dali.fn.transforms.scale(scale=[2,2,2])), u)
    pipe.build()
    matrices, offsets = pipe.run()
    assert len(matrices) == batch_size
    for i in range(len(matrices)):
        offset = offsets.at(i)
        matrix = matrices.at(i)
        assert offset.shape == (3,)
        for j in range(3):
            assert offset[j] >= 1 and offset[j] < 2  # check that it's not all zeros or sth
        mtx = np.float32([[0,-1, 0],
                          [1, 0, 0],
                          [0, 0, -1]])
        T = offset[:,np.newaxis]  # convert to a columnn
        T = np.dot(mtx, T)
        mtx *= 2
        assert np.allclose(matrix, np.concatenate([mtx, T], axis=1), rtol=1e-5, atol=1e-6)

