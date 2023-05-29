# from nvidia.dali.auto_aug.core import select
from nvidia.dali import fn, pipeline_def, types
import numpy as np

from nvidia.dali._autograph.utils import ag_logging as logging

# logging.set_verbosity(10, True)

def _select0(op_range_lo: int, op_range_hi: int, ops, selected_op_idx, op_args, op_kwargs):
    assert op_range_lo <= op_range_hi
    if op_range_lo == op_range_hi:
        return ops[op_range_lo](*op_args, **op_kwargs)
    mid = (op_range_lo + op_range_hi) // 2
    if selected_op_idx <= mid:
        ret = _select0(op_range_lo, mid, ops, selected_op_idx, op_args, op_kwargs)
    else:
        ret = _select0(mid + 1, op_range_hi, ops, selected_op_idx, op_args, op_kwargs)
    return ret

def _select1(op_range_lo: int, op_range_hi: int, ops, selected_op_idx, op_args, op_kwargs):
    assert op_range_lo <= op_range_hi
    if op_range_lo == op_range_hi:
        return ops[op_range_lo](*op_args, **op_kwargs)
    mid = (op_range_lo + op_range_hi) // 2
    if selected_op_idx <= mid:
        return _select1(op_range_lo, mid, ops, selected_op_idx, op_args, op_kwargs)
    else:
        return _select1(mid + 1, op_range_hi, ops, selected_op_idx, op_args, op_kwargs)

def _select2(op_range_lo: int, op_range_hi: int, ops, selected_op_idx, op_args, op_kwargs):
    assert op_range_lo <= op_range_hi
    if op_range_lo == op_range_hi:
        return ops[op_range_lo](*op_args, **op_kwargs)
    mid = (op_range_lo + op_range_hi) // 2
    if selected_op_idx <= mid:
        a, b = _select2(op_range_lo, mid, ops, selected_op_idx, op_args, op_kwargs)
    else:
        a, b = _select2(mid + 1, op_range_hi, ops, selected_op_idx, op_args, op_kwargs)
    return a, b


def select(ops, selected_op_idx, *op_args, unpacking_select=False, **op_kwargs):
    if unpacking_select == 2:
        return _select2(0, len(ops) - 1, ops, selected_op_idx, op_args, op_kwargs)
    elif unpacking_select == 1:
        return _select1(0, len(ops) - 1, ops, selected_op_idx, op_args, op_kwargs)
    else:
        return _select0(0, len(ops) - 1, ops, selected_op_idx, op_args, op_kwargs)


def rotate(image, label):
    image = fn.rotate(image, angle=42)
    return image, label

def color(image, label):
    image = fn.color_twist(image, saturation=0)
    return image, label

def wrapped_rotate(image, label):
    return rotate(image, label)

def source_cbk(source_info):
    return np.full((200, 300, 3), 42, dtype=np.uint8), np.array(1)


@pipeline_def(enable_conditionals=True, num_threads=4, batch_size=8, device_id=0)
def pipeline(unpacking_select):
    image, label = fn.external_source(source=source_cbk, num_outputs=2, batch=False, layout=("HWC", ""))
    image = types.Constant(np.full((200, 300, 3), 42, dtype=np.uint8), device="cpu")
    label =types.Constant(np.array(1), device="cpu")
    image = fn.resize(image, size=(400, 600))
    ops = [rotate, color]
    op_idx = fn.random.uniform(values=list(range(len(ops))))
    image, label = select(ops, op_idx, image=image, label=label, unpacking_select=unpacking_select)
    return image, label


def test_2():
    pipe = pipeline(2)
    pipe.build()
    pipe.run()

def test_1():
    pipe = pipeline(1)
    pipe.build()
    pipe.run()

def test_0():
    pipe = pipeline(0)
    pipe.build()
    pipe.run()

