from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import torch.utils.dlpack as torch_dl
import numpy as np
import os
import random
import cupy

BATCH_SIZE = 4
THREADS = 4
SEED = int(random.random() * (1 << 32))
DEVICE_ID = 0
ITERS = 5
test_data_root = os.environ['DALI_EXTRA_PATH']
images_dir = os.path.join(test_data_root, 'db', 'single', 'jpeg')


class LoadingPipeline(Pipeline):
    def __init__(self, device):
        super(LoadingPipeline, self).__init__(BATCH_SIZE, THREADS, DEVICE_ID, SEED)
        self.input = ops.FileReader(file_root=images_dir)
        self.decode = ops.ImageDecoder(output_type=types.RGB)
        self.device = device

    def define_graph(self):
        jpegs, labels = self.input()
        return self.decode(jpegs) if self.device == 'cpu' else self.decode(jpegs).gpu()


def test_cpu_tensor():
    pipe1 = LoadingPipeline('cpu')
    pipe2 = LoadingPipeline('cpu')
    pipe1.build()
    pipe2.build()
    for iter in range(ITERS):
        batch1, = pipe1.run()
        batch2, = pipe2.run()
        dl_tensors = batch1.as_dlpack()
        arrays1 = [torch_dl.from_dlpack(dl).numpy() for dl in dl_tensors]
        arrays2 = [batch2.at(i) for i in range(BATCH_SIZE)]
        for i in range(BATCH_SIZE):
            assert np.array_equal(arrays1[i], arrays2[i])


def test_gpu_tensor():
    pipe1 = LoadingPipeline('gpu')
    pipe2 = LoadingPipeline('gpu')
    pipe1.build()
    pipe2.build()
    for iter in range(ITERS):
        batch1, = pipe1.run()
        batch2, = pipe2.run()
        dl_tensors = batch1.as_dlpack()
        batch2_cpu = batch2.as_cpu()
        arrays1 = [torch_dl.from_dlpack(dl).cpu().numpy() for dl in dl_tensors]
        arrays2 = [batch2_cpu.at(i) for i in range(BATCH_SIZE)]
        for i in range(BATCH_SIZE):
            assert np.array_equal(arrays1[i], arrays2[i])


def test_convert_to_cupy():
    pipe1 = LoadingPipeline('gpu')
    pipe2 = LoadingPipeline('gpu')
    pipe1.build()
    pipe2.build()
    for iter in range(ITERS):
        batch1, = pipe1.run()
        batch2, = pipe2.run()
        dl_tensors = batch1.as_dlpack()
        batch2_cpu = batch2.as_cpu()
        arrays1 = [cupy.asnumpy(cupy.fromDlpack(dl)) for dl in dl_tensors]
        arrays2 = [batch2_cpu.at(i) for i in range(BATCH_SIZE)]
        for i in range(BATCH_SIZE):
            assert np.array_equal(arrays1[i], arrays2[i])

