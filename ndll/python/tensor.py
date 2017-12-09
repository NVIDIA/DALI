import ndll.backend as b
from ndll.backend import TensorCPU
from ndll.backend import TensorListCPU

class TensorReference(object):
    def __init__(self, name, device = "cpu", source = None):
        self.name = name
        self.device = device
        self.source = source
