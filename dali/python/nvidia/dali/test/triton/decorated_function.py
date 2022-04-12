import nvidia.dali as dali


@dali.triton.autoserialize
@dali.pipeline_def
def func_under_test():
    return 42
