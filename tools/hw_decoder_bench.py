import nvidia.dali as dali
from nvidia.dali.pipeline import Pipeline
from PIL import Image
import argparse
import time

parser = argparse.ArgumentParser(description='DALI HW decoder benchmark')
parser.add_argument('-b', dest="batch_size", help="batch size", default=1, type=int)
parser.add_argument('-dev', dest="device_id", help="device id", default=0, type=int)
parser.add_argument('-w', dest="warmup_iterations", help="warmup iterations", default=0, type=int)
parser.add_argument('-t', dest="total_images", help="total images", default=100, type=int)
parser.add_argument('-j', dest="num_threads", help="num_threads", default=1, type=int)
parser.add_argument('-i', dest="images_dir", help="images dir")
parser.add_argument('--hw_load', dest="hw_load", help="HW decoder workload (e.g. 0.66 means 66% of the batch)", default=0.65, type=float)
parser.add_argument('--width_hint', dest="width_hint", default=0, type=int)
parser.add_argument('--height_hint', dest="height_hint", default=0, type=int)
parser.add_argument('-dec', dest="decoder")

args = parser.parse_args()

class SamplePipeline(Pipeline):
    def __init__(self, batch_size=args.batch_size, num_threads=args.num_threads, device_id=args.device_id):
        super(SamplePipeline, self).__init__(batch_size, num_threads, device_id, seed=0)
        self.input = dali.ops.readers.File(file_root = args.images_dir)
        self.decode = dali.ops.ImageDecoder(
            device = 'mixed',
            output_type = dali.types.RGB,
            hw_decoder_load=args.hw_load,
            preallocate_width_hint=args.width_hint,
            preallocate_height_hint=args.height_hint)
    def define_graph(self):
        jpegs, _ = self.input()
        images = self.decode(jpegs)
        return images

pipe = SamplePipeline()
pipe.build()

for iteration in range(args.warmup_iterations):
    output = pipe.run()
print("Warmup finished")

start = time.time()
test_iterations = args.total_images // args.batch_size

print('Test iterations: ', test_iterations)
for iteration in range(test_iterations):
    output = pipe.run()
end = time.time()
total_time = end - start

print(test_iterations * args.batch_size / total_time, 'fps')
