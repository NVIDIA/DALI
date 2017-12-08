import ndll.backend as b


class TensorReference(object):
    def __init__(self, name, device = "cpu", source = None):
        self.name = name
        self.device = device
        self.source = source
