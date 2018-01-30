import mxnet as mx
import ndll
import numpy as np
from ndll.pipeline import Pipeline
import ndll.ops as ops
import ndll.types as types
import ndll.tfrecord as tfrec
import numpy as np
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt

tfrecord = "/data/imagenet/train-val-tfrecord-480/train-00001-of-01024"
tfrecord_idx = "/outputs/train-00001-of-01024.idx"

class HybridPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, pipelined = True, async = True):
        super(HybridPipe, self).__init__(batch_size, num_threads, device_id, pipelined, async)
        self.input = ops.TFRecordReader(path = tfrecord, 
                                        index_path = tfrecord_idx,
                                        features = {"image/encoded" :         tfrec.FixedLenFeature((), tfrec.string, ""),
                                         'image/class/label':      tfrec.FixedLenFeature([1], tfrec.int64,  -1),
                                         'image/class/text':       tfrec.FixedLenFeature([ ], tfrec.string, ''),
                                         'image/object/bbox/xmin': tfrec.VarLenFeature(tfrec.float32, 0.0),
                                         'image/object/bbox/ymin': tfrec.VarLenFeature(tfrec.float32, 0.0),
                                         'image/object/bbox/xmax': tfrec.VarLenFeature(tfrec.float32, 0.0),
                                         'image/object/bbox/ymax': tfrec.VarLenFeature(tfrec.float32, 0.0)})
        self.huffman = ops.HuffmanDecoder()
        self.idct = ops.DCTQuantInv(device = "gpu", output_type = types.RGB)
        self.resize = ops.Resize(device = "gpu", random_resize = True,
                                 resize_a = 256, resize_b = 480,
                                 image_type = types.RGB,
                                 interp_type = types.INTERP_LINEAR)
        self.cmnp = ops.CropMirrorNormalizePermute(device = "gpu",
                                                   output_type = types.FLOAT,
                                                   random_crop = True,
                                                   crop_h = 224,
                                                   crop_w = 224,
                                                   image_type = types.RGB,
                                                   mean = [128., 128., 128.],
                                                   std = [1., 1., 1.])
        self.iter = 0

    def define_graph(self):
        inputs = self.input(name="Reader")
        dct_coeff, jpeg_meta = self.huffman(inputs["image/encoded"])
        images = self.idct(dct_coeff.gpu(), jpeg_meta)
        images = self.resize(images)
        output = self.cmnp(images)
        return (output, inputs["image/class/label"], inputs["image/class/text"])

    def iter_setup(self):
        pass

def run_benchmarks(PipeType, args):
    print("Running Benchmarks For {}".format(PipeType.__name__))
    for executor in args.executors:
        pipelined = executor > 0
        async = executor > 1
        for batch_size in args.batch_sizes:
            for num_threads in args.thread_counts:
                pipe = PipeType(batch_size, num_threads, 0, pipelined, async)
                pipe.build()
                start_time = timer()
                for i in range(args.num_iters):
                    pipe.run()

                total_time = timer() - start_time
                print("{}/{}/{}/{}: FPS={}"
                      .format(PipeType.__name__,  executor, batch_size, num_threads,
                              float(batch_size * args.num_iters) / total_time))
class Struct:
    def __init__(self, **entries): self.__dict__.update(entries)

def test():
    args = Struct(batch_sizes=[128], thread_counts=[1,2,3,4], executors=[2], num_iters=100)
    pipe_types = [HybridPipe]
    for PipeType in pipe_types:
        run_benchmarks(PipeType, args)

pipe = HybridPipe(batch_size=128, num_threads=4, device_id = 0, pipelined = True, async = True)

pipe.build()
pipe_out = pipe.run()
t1 = pipe_out[0].as_tensor()
t2 = pipe_out[1].as_tensor()
print(t1.shape())
print(t2.shape())
a = mx.nd.zeros((128, 3, 224, 224), mx.gpu(0))
import ctypes
ptr = ctypes.c_void_p()
mx.base._LIB.MXNDArrayGetData(ctypes.byref(a.handle), ctypes.byref(ptr))
t1.copy_to_external(ptr)
print(a)
