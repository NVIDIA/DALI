import nvidia.dali as dali


@dali.triton.autoserialize
@dali.pipeline_def(max_batch_size=1, num_threads=1, device_id=0)
def func_under_test():
    return 42


@dali.triton.autoserialize
@dali.pipeline_def(max_batch_size=1, num_threads=1, device_id=0)
def func_that_shouldnt_be_here():
    return 42
