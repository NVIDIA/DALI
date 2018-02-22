from ndll.pipeline import Pipeline
import ndll.ops as ops
import ndll.types as types
import ndll.tfrecord as tfrec
import numpy as np
from timeit import default_timer as timer
import numpy as np

def test_tensor_multiple_uses:
    db_folder = "/data/imagenet/train-lmdb-256x256"
    batch_size = 128
    class HybridPipe(Pipeline):
        def __init__(self, batch_size, num_threads, device_id, num_gpus, pipelined = True, async = True):
            super(HybridPipe, self).__init__(batch_size, num_threads, device_id, pipelined, async)
            self.input = ops.CaffeReader(path = db_folder, shard_id = device_id, num_shards = num_gpus)
            self.decode = ops.TJPGDecoder(device = "cpu", output_type = types.RGB)
            self.dump_cpu = ops.DumpImage(device = "cpu", suffix = "cpu")
            self.dump_gpu = ops.DumpImage(device = "gpu", suffix = "gpu")
            
        def define_graph(self):
            inputs, labels = self.input(name="Reader")
            images = self.decode(inputs)
            images_cpu = self.dump_cpu(images)
            images_gpu = self.dump_gpu(images.gpu())
            return (images, images_cpu, images_gpu)

        def iter_setup(self):
            pass

    pipe = HybridPipe(batch_size=batch_size, num_threads=1, device_id = 0, num_gpus = 1, pipelined = True, async = True)
    pipe.build()
    out = pipe.run()
    assert(out[0].is_dense_tensor())
    assert(out[1].is_dense_tensor())
    assert(out[2].is_dense_tensor())
    assert(out[0].as_tensor().shape() == out[1].as_tensor().shape())
    assert(out[0].as_tensor().shape() == out[2].as_tensor().shape())
    a_raw = out[0]
    a_cpu = out[1]
    a_gpu = out[2].asCPU()
    for i in range(batch_size):
        t_raw = a_raw.at(i)
        t_cpu = a_cpu.at(i)
        assert(np.sum(np.abs(t_cpu - t_raw)) == 0)
        t_cpu = a_cpu.at(i)
        t_gpu = a_gpu.at(i)
        assert(np.sum(np.abs(t_cpu - t_gpu)) == 0)
