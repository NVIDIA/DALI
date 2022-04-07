import nvidia.dali as dali
from pipeline_stub import PipelineStub

@dali.triton.autoserialize
def func_under_test():
    return PipelineStub()
