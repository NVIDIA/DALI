from ndll.pipeline import Pipeline
import ndll.ops as ops
import ndll.types as types
import numpy as np
from timeit import default_timer as timer

image_folder = "./benchmark_images"

def read_jpegs(folder):
    with open(folder + "/image_list.txt", 'r') as file:
        files = [line.rstrip() for line in file]

    images = []
    for fname in files:
        f = open(image_folder + "/" + fname, 'rb')
        images.append(np.fromstring(f.read(), dtype = np.uint8))
    return images

def make_batch(size):
    data = read_jpegs(image_folder)
    return [data[i % len(data)] for i in range(size)]

class DataPipe(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(DataPipe, self).__init__(batch_size, num_threads, device_id, True, True)
        self.input = ops.ExternalSource()
        self.decode = ops.TJPGDecoder(output_type = types.RGB)
        self.rcm = ops.FastResizeCropMirror(random_resize = True,
                                            resize_a = 256,
                                            resize_b = 480,
                                            random_crop = True,
                                            crop_h = 224,
                                            crop_w = 224)
        self.np = ops.NormalizePermute(device = "gpu",
                                       output_type = types.FLOAT16,
                                       mean = [128., 128., 128.],
                                       std = [1., 1., 1.],
                                       height = 224,
                                       width = 224,
                                       channels = 3)
        self.iter = 0

    def define_graph(self):
        self.jpegs = self.input()
        images = self.decode(self.jpegs)
        resized = self.rcm(images)
        output = self.np(resized.gpu())
        return output

    def iter_setup(self):
        if self.iter == 0:
            raw_data = make_batch(self.batch_size)
            self.feed_input(self.jpegs, raw_data)
            self.iter += 1

batch_size = 128
num_threads = 4

pipe = DataPipe(batch_size,
                num_threads,
                0)

pipe.build()
max_iters = 100
start_time = timer()
for i in range(max_iters):
    pipe.run()

total_time = timer() - start_time
print("Images/second: {}".format(float(batch_size * max_iters) / total_time))
