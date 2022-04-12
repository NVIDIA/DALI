import nvidia.dali as dali


@dali.triton.autoserialize
@dali.pipeline_def(batch_size=1, num_threads=1, device_id=0)
def func_under_test():
    return 42
