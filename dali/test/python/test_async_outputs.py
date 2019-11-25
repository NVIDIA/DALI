import nvidia.dali.ops as ops
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.types as types
import torch.utils.dlpack as torch_dlpack
import torch
import os
import random
import numpy
from PIL import Image


test_data_root = os.environ['DALI_EXTRA_PATH']
images_dir = os.path.join(test_data_root, 'db', 'single', 'jpeg')


def random_seed():
    return int(random.random() * (1 << 32))


DEVICE_ID = 0
BATCH_SIZE = 20
ITERS = 64
SEED = random_seed()
NUM_WORKERS = 6


class TestPipeline(Pipeline):
    def __init__(self):
        super(TestPipeline, self).__init__(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, seed=SEED)
        self.input = ops.FileReader(file_root=images_dir)
        self.decode = ops.ImageDecoder(device='mixed', output_type=types.RGB)
        self.resize = ops.Resize(resize_x=1000, resize_y=1000, device='gpu')
        self.flip = ops.Flip(device='gpu')

    def define_graph(self):
        jpegs, labels = self.input()
        images = self.decode(jpegs)
        return self.flip(self.resize(images))


def to_torch_tensors(tensor_list):
    dltensor_list = tensor_list.as_dlpack()
    return [torch_dlpack.from_dlpack(t) for t in dltensor_list]


torch_stream = torch.cuda.Stream()


def test_async_output():
    pipe = TestPipeline()
    pipe.build()
    results = []
    for i in range(ITERS):
        pipe.schedule_run()
        [image], output_event = pipe.async_outputs()
        tensors = to_torch_tensors(image)
        with torch.cuda.stream(torch_stream):
            torch_stream.wait_event(output_event)
            results.append([t.cpu().numpy() for t in tensors])
        pipe.async_release_outputs(torch_stream)

    sync_pipe = TestPipeline()
    sync_pipe.build()
    for i in range(ITERS):
        [image] = sync_pipe.run()
        data = image.as_cpu()
        print(i)
        for s in range(BATCH_SIZE):
            print("s " + str(s))
            if not numpy.array_equal(results[i][s], data.at(s)):
                Image.fromarray(results[i][s]).save('ver.jpg')
                Image.fromarray(data.at(s)).save('ref.jpg')
                assert False
