# Copyright (c) 2017-2018, NVIDIA CORPORATION. All rights reserved.
#pylint: disable=no-name-in-module, unused-import
import nvidia.dali.backend
from nvidia.dali.backend import TensorCPU
from nvidia.dali.backend import TensorListCPU

class TensorReference(object):
    def __init__(self, name, device="cpu", source=None):
        self.name = name
        self.device = device
        self.source = source

    # Note: Regardless of whether we want the cpu or gpu version
    # of a tensor, we keep the source argument the same so that
    # the pipeline can backtrack through the user-defined graph
    def cpu(self):
        return TensorReference(self.name, "cpu", self.source)

    def gpu(self):
        return TensorReference(self.name, "gpu", self.source)
