import nvidia.dali as dali


@dali.plugin.triton.autoserialize
def func_under_test():
    return 42