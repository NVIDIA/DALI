import ndll.backend as b

# Python format documentation
# Python style guide
# Auto-expose ndll.ops [x]
# ndll.tensor module w/ TensorReference, TensorCPU, TensorGPU
class Pipeline(object):
    def __init__(self, batch_size, num_threads, device_id, queue_depth = 2,
                 bytes_per_sample = 0, set_affinity = False, max_streams = -1):
        self._pipe = b.Pipeline(batch_size,
                                num_threads,
                                device_id,
                                queue_depth,
                                bytes_per_sample,
                                set_affinity,
                                max_streams)

    def build():
        outputs = build_graph()
        for i in outputs:
            print(i)
    
    # defined by the user to construct their graph of operations.
    # this returns a list of output TensorReferences that we can
    # trace back to add them to the graph
    def build_graph():
        raise NotImplementedError

class TensorReference(object):
    def __init__(self, name, device = "cpu", source = None):
        self.name = name
        self.device = device
        self.parent = None
    
