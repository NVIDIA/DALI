import nvidia.dali as dali
from pipeline_stub import PipelineStub


@dali.triton.autoserialize
def func_under_test():
    return PipelineStub()


@dali.triton.autoserialize
def func_that_shouldnt_be_here():
    return PipelineStub()
