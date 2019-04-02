from __future__ import print_function
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy
import argparse
import random
import torchvision.transforms as transforms
from PIL import Image

images_dirs = [
    "/data/imagenet/val-jpeg"
]


def resize(image):
    res = transforms.Resize((300, 300))
    return numpy.array(res(Image.fromarray(image)))


class CommonPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, _seed, image_dir):
        super(CommonPipeline, self).__init__(batch_size, num_threads, device_id, seed=_seed, exec_async=False,
                                             exec_pipelined=False)
        self.input = ops.FileReader(file_root=image_dir)
        self.decode = ops.HostDecoder(output_type=types.RGB)
        self.resize = ops.PythonFunction(function_id=id(resize))

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
        self.python_function = ops.PythonFunction(function_id=id(function))

    def define_graph(self):
        images, labels = self.load()
        processed = self.python_function(images)
        return processed


class FlippingPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, seed, image_dir):
        super(FlippingPipeline, self).__init__(batch_size, num_threads, device_id, seed, image_dir)
        self.flip = ops.Flip(horizontal=1)

    def define_graph(self):
        images, labels = self.load()
        flipped = self.flip(images)
        return flipped


DEVICE_ID = 0


def random_seed():
    return int(random.random() * (1 << 32))


def make_parser():
    _parser = argparse.ArgumentParser(description='Python function operator test')
    _parser.add_argument(
        '-i', '--iters', default=32, type=int, metavar='N',
        help='number of iterations to run (default: whole dataset)')
    _parser.add_argument(
        '-s', '--seed', default=random_seed(), type=int, metavar='N',
        help='seed for random ops (default: random seed)')
    _parser.add_argument(
        '-w', '--num_workers', default=3, type=int, metavar='N',
        help='number of worker threads (default: %(default)s)')
    _parser.add_argument(
        '-b', '--batch_size', default=1, type=int, metavar='N',
        help='image batch size (default: %(default)s)')
    return _parser


def run_test(func, name):
    pipe = BasicPipeline(args.batch_size, args.num_workers, DEVICE_ID, args.seed, images_dirs[0])
    pyfunc_pipe = PythonOperatorPipeline(args.batch_size, args.num_workers, DEVICE_ID, args.seed, images_dirs[0], func)
    pipe.build()
    pyfunc_pipe.build()
    print('test: ' + name)
    for it in range(args.iters):
        preprocessed_output, = pipe.run()
        output, = pyfunc_pipe.run()
        for i in range(len(output)):
            assert numpy.array_equal(output.at(i), func(preprocessed_output.at(i)))
    print('done')


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


def flipping_test(args):
    dali_flip = FlippingPipeline(args.batch_size, args.num_workers, DEVICE_ID, args.seed, images_dirs[0])
    numpy_flip = PythonOperatorPipeline(args.batch_size, args.num_workers, DEVICE_ID, args.seed, images_dirs[0], flip)
    dali_flip.build()
    numpy_flip.build()
    print('test: flip')
    for it in range(args.iters):
        numpy_output, = numpy_flip.run()
        dali_output, = dali_flip.run()
        for i in range(len(numpy_output)):
            assert numpy.array_equal(numpy_output.at(i), dali_output.at(i))
    print('done')


if __name__ == '__main__':
    parser = make_parser()
    args = parser.parse_args()
    flipping_test(args)
    run_test(one_channel_normalize, "one channel normalized")
    run_test(channels_mean, "channels mean")
    run_test(bias, "bias")
