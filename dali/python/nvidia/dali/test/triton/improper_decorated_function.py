from nvidia.dali.plugin.triton import autoserialize


@autoserialize
def func_under_test():
    return 42
