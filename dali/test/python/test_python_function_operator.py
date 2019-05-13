from __future__ import print_function
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.edge import EdgeReference
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy
import random
import torchvision.transforms as transforms
from PIL import Image
import os

test_data_root = os.environ['DALI_EXTRA_PATH']
images_dir = os.path.join(test_data_root, 'db', 'single', 'jpeg')

def resize(image):
    res = transforms.Resize((300, 300))
    return numpy.array(res(Image.fromarray(image)))


class CommonPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, _seed, image_dir):
        super(CommonPipeline, self).__init__(batch_size, num_threads, device_id, seed=_seed, exec_async=False,
                                             exec_pipelined=False)
        self.input = ops.FileReader(file_root=image_dir)
        self.decode = ops.HostDecoder(output_type=types.RGB)
        self.resize = ops.PythonFunction(function=resize)

    def load(self):
        jpegs, labels = self.input()
        decoded = self.decode(jpegs)
        resized = self.resize(decoded)
        return resized, labels

    def define_graph(self):
        pass


class BasicPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, seed, image_dir):
        super(BasicPipeline, self).__init__(batch_size, num_threads, device_id, seed, image_dir)

    def define_graph(self):
        images, labels = self.load()
        return images


class PythonOperatorPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, seed, image_dir, function):
        super(PythonOperatorPipeline, self).__init__(batch_size, num_threads, device_id, seed, image_dir)
        self.python_function = ops.PythonFunction(function=function)

    def define_graph(self):
        images, labels = self.load()
        processed = self.python_function(images)
        assert isinstance(processed, EdgeReference)
        return processed


class FlippingPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, seed, image_dir):
        super(FlippingPipeline, self).__init__(batch_size, num_threads, device_id, seed, image_dir)
        self.flip = ops.Flip(horizontal=1)

    def define_graph(self):
        images, labels = self.load()
        flipped = self.flip(images)
        return flipped


def random_seed():
    return int(random.random() * (1 << 32))


DEVICE_ID = 0
BATCH_SIZE = 8
ITERS = 64
SEED = random_seed()
NUM_WORKERS = 6


def run_case(func):
    pipe = BasicPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir)
    pyfunc_pipe = PythonOperatorPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, func)
    pipe.build()
    pyfunc_pipe.build()
    for it in range(ITERS):
        preprocessed_output, = pipe.run()
        output, = pyfunc_pipe.run()
        for i in range(len(output)):
            assert numpy.array_equal(output.at(i), func(preprocessed_output.at(i)))


def one_channel_normalize(image):
    return image[:, :, 1] / 255.


def channels_mean(image):
    r = numpy.mean(image[:, :, 0])
    g = numpy.mean(image[:, :, 1])
    b = numpy.mean(image[:, :, 2])
    return numpy.array([r, g, b])


def bias(image):
    return numpy.array(image > 127, dtype=numpy.bool)


def flip(image):
    return numpy.fliplr(image)


def test_python_operator_one_channel_normalize():
    run_case(one_channel_normalize)

def test_python_operator_channels_mean():
    run_case(channels_mean)


def test_python_operator_bias():
    run_case(bias)


def test_python_operator_flip():
    dali_flip = FlippingPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir)
    numpy_flip = PythonOperatorPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, flip)
    dali_flip.build()
    numpy_flip.build()
    for it in range(ITERS):
        numpy_output, = numpy_flip.run()
        dali_output, = dali_flip.run()
        for i in range(len(numpy_output)):
            assert numpy.array_equal(numpy_output.at(i), dali_output.at(i))
