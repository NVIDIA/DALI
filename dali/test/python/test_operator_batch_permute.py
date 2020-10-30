import nvidia.dali as dali
import nvidia.dali.fn as fn
from nvidia.dali.pipeline import Pipeline
import numpy as np
from test_utils import check_batch

def _test_permutation_generator(allow_repetitions, no_fixed):
    batch_size = 10
    pipe = Pipeline(batch_size, 1, None)
    perm = fn.batch_permutation(allow_repetitions=allow_repetitions, no_fixed_points=no_fixed)
    pipe.set_outputs(perm)

    pipe.build()
    for iter in range(100):
        idxs, = pipe.run()
        for i in range(batch_size):
            assert idxs.at(i).shape == ()
        idxs = [int(idxs.at(i)) for i in range(batch_size)]
        if allow_repetitions:
            assert all(x >= 0 and x < batch_size for x in idxs)
        else:
            assert list(sorted(idxs)) == list(range(batch_size))

        if no_fixed:
            assert all(x != i for i, x in enumerate(idxs))


def test_permutation_generator():
    for allow_repetitions in [None, False, True]:
        for no_fixed in [None, False, True]:
            yield _test_permutation_generator, allow_repetitions, no_fixed

def random_sample():
    shape = np.random.randint(1, 50, [3])
    return np.random.randint(-1000000, 1000000, shape)

def gen_data(batch_size, type):
    return [random_sample().astype(type) for _ in range(batch_size)]

def _test_permute_batch(device, type):
    batch_size = 10
    pipe = Pipeline(batch_size, 4, 0)
    data = fn.external_source(source=lambda: gen_data(batch_size, type), device=device, layout="abc")
    perm = fn.batch_permutation()
    pipe.set_outputs(data, fn.permute_batch(data, indices=perm), perm)
    pipe.build()

    for i in range(10):
        orig, permuted, idxs = pipe.run()
        idxs = [int(idxs.at(i)) for i in range(batch_size)]
        if isinstance(orig, dali.backend.TensorListGPU):
            orig = orig.as_cpu()
        ref = [orig.at(idx) for idx in idxs]
        check_batch(permuted, ref, len(ref), 0, 0, "abc")

def test_permute_batch():
    for type in [np.uint8, np.int16, np.uint32, np.int64, np.float32]:
        for device in ["cpu", "gpu"]:
            yield _test_permute_batch, device, type
