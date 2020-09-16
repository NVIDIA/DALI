import numpy as np
import nvidia.dali as dali
import nvidia.dali.fn as fn
from test_utils import check_batch, dali_type

def make_param(kind, shape):
    if kind == "input":
        return fn.uniform(range=(0, 1), shape=shape)
    elif kind == "scalar input":
        return fn.reshape(fn.uniform(range=(0, 1)), shape=[])
    elif kind == "vector":
        return np.random.rand(*shape).astype(np.float32)
    elif kind == "scalar":
        return np.random.rand()
    else:
        return None

def clip(value, type = None):
    try:
        info = np.iinfo(type)
        return np.clip(value, info.min, info.max)
    except:
        return value

def make_data_batch(batch_size, in_dim, type):
    np.random.seed(1234)
    batch = []
    lo = 0
    hi = 1
    try:
        info = np.iinfo(type)
        lo = max(info.min / 2, -1000000)
        hi = min(info.max / 2,  1000000)
    except:
        pass

    for i in range(batch_size):
        batch.append((np.random.rand(np.random.randint(1, 10), in_dim)*(hi-lo) + lo).astype(type))
    return batch

def get_data_source(batch_size, in_dim, type):
    return lambda: make_data_batch(batch_size, in_dim, type)

def _run_test(device, batch_size, out_dim, in_dim, in_dtype, out_dtype, M_kind, T_kind):
    pipe = dali.pipeline.Pipeline(batch_size=batch_size, num_threads=2, device_id=0, seed=1234)
    with pipe:
        X = fn.external_source(source=get_data_source(batch_size, in_dim, in_dtype), device=device)
        M = make_param(M_kind, [out_dim, in_dim])
        T = make_param(T_kind, [out_dim])
        Y = fn.coord_transform(X,
                               M = M.flatten().tolist() if isinstance(M, np.ndarray) else M,
                               T = T.flatten().tolist() if isinstance(T, np.ndarray) else T,
                               dtype = dali_type(out_dtype)
                               )
        if M is None:
            M = 1
        if T is None:
            T = 0

        M, T = (x if isinstance(x, dali.data_node.DataNode) else dali.types.Constant(x, dtype=dali.types.FLOAT) for x in (M, T))

        pipe.set_outputs(X, M, T, Y)

    pipe.build()
    for iter in range(3):
        outputs = pipe.run()
        outputs = [x.as_cpu() if hasattr(x, "as_cpu") else x for x in outputs]
        ref = []
        scale = 1
        for idx in range(batch_size):
            X = outputs[0].at(idx)
            M = outputs[1].at(idx)
            T = outputs[2].at(idx)

            if M.size == 1:
               ref.append(X.astype(np.float32) * M + T)
            else:
               ref.append(np.matmul(X.astype(np.float32), M.transpose()) + T)
            scale = max(scale, np.max(np.abs(ref[-1])) - np.min(np.abs(ref[-1])))
        avg = 1e-6 * scale
        eps = 1e-4 * scale
        if out_dtype != np.float32:  # headroom for rounding
            avg += 0.33
            eps += 0.5
        check_batch(outputs[3], ref, batch_size, eps, eps, compare_layouts=False)


def test_all():
    for device in ["cpu", "gpu"]:
        for M_kind in [None, "vector", "scalar", "input", "scalar input"]:
            for T_kind in [None, "vector", "scalar", "input", "scalar input"]:
                for batch_size in [1,3]:
                    yield _run_test, device, batch_size, 3, 3, np.float32, np.float32, M_kind, T_kind

    for device in ["cpu", "gpu"]:
        for in_dtype in [np.uint8, np.uint16, np.int16, np.int32, np.float32]:
            for out_dtype in set([in_dtype, np.float32]):
                for batch_size in [1,8]:
                    yield _run_test, device, batch_size, 3, 3, in_dtype, out_dtype, "input", "input"

    for device in ["cpu", "gpu"]:
        for M_kind in ["input", "scalar", None]:
            for in_dim in [1,2,3,4,5,6]:
                out_dims = [1,2,3,4,5,6] if M_kind == "vector" or M_kind == "input" else [in_dim]
                for out_dim in out_dims:
                    yield _run_test, device, 2, out_dim, in_dim, np.float32, np.float32, M_kind, "vector"
