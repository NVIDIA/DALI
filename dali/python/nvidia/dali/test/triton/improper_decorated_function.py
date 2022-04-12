import nvidia.dali as dali


@dali.triton.autoserialize
def func_under_test():
    return 42