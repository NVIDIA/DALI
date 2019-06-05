from __future__ import print_function
from nvidia.dali.pipeline import Pipeline
from nvidia.dali.edge import EdgeReference
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import numpy
import random
from PIL import Image
import os

test_data_root = os.environ['DALI_EXTRA_PATH']
images_dir = os.path.join(test_data_root, 'db', 'single', 'jpeg')


def resize(image):
    return numpy.array(Image.fromarray(image).resize((300, 300)))


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


class TwoOutputsPythonOperatorPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, seed, image_dir, function):
        super(TwoOutputsPythonOperatorPipeline, self).__init__(batch_size, num_threads,
                                                               device_id, seed, image_dir)
        self.python_function = ops.PythonFunction(function=function, num_outputs=2)

    def define_graph(self):
        images, labels = self.load()
        out1, out2 = self.python_function(images)
        assert isinstance(out1, EdgeReference)
        assert isinstance(out2, EdgeReference)
        return out1, out2


class MultiInputMultiOutputPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, seed, image_dir, function):
        super(MultiInputMultiOutputPipeline, self).__init__(batch_size, num_threads,
                                                            device_id, seed, image_dir)
        self.python_function = ops.PythonFunction(function=function, num_outputs=3)

    def define_graph(self):
        images1, labels1 = self.load()
        images2, labels2 = self.load()
        out1, out2, out3 = self.python_function(images1, images2)
        assert isinstance(out1, EdgeReference)
        assert isinstance(out2, EdgeReference)
        assert isinstance(out3, EdgeReference)
        return out1, out2, out3


class DoubleLoadPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, seed, image_dir):
        super(DoubleLoadPipeline, self).__init__(batch_size, num_threads,
                                                 device_id, seed, image_dir)

    def define_graph(self):
        images1, labels1 = self.load()
        images2, labels2 = self.load()
        return images1, images2


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

def invalid_function(image):
    return img

def test_python_operator_invalid_function():
    invalid_pipe = PythonOperatorPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir,
                                          invalid_function)
    invalid_pipe.build()
    try:
        invalid_pipe.run()
    except Exception as e:
        print(e)
        return
    raise Exception('Should not pass')


def split_red_blue(image):
    return image[:, :, 0], image[:, :, 2]


def mixed_types(image):
    return bias(image), one_channel_normalize(image)


def run_two_outputs(func):
    pipe = BasicPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir)
    pyfunc_pipe = TwoOutputsPythonOperatorPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED,
                                                   images_dir, func)
    pipe.build()
    pyfunc_pipe.build()
    for it in range(ITERS):
        preprocessed_output, = pipe.run()
        output1, output2 = pyfunc_pipe.run()
        for i in range(len(output1)):
            pro1, pro2 = func(preprocessed_output.at(i))
            assert numpy.array_equal(output1.at(i), pro1)
            assert numpy.array_equal(output2.at(i), pro2)


def test_split():
    run_two_outputs(split_red_blue)


def test_mixed_types():
    run_two_outputs(mixed_types)


def run_multi_input_multi_output(func):
    pipe = DoubleLoadPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir)
    pyfunc_pipe = MultiInputMultiOutputPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED,
                                                images_dir, func)
    pipe.build()
    pyfunc_pipe.build()
    for it in range(ITERS):
        preprocessed_output1, preprocessed_output2 = pipe.run()
        out1, out2, out3 = pyfunc_pipe.run()
        for i in range(len(out1)):
            pro1, pro2, pro3 = func(preprocessed_output1.at(i), preprocessed_output2.at(i))
            assert numpy.array_equal(out1.at(i), pro1)
            assert numpy.array_equal(out2.at(i), pro2)
            assert numpy.array_equal(out3.at(i), pro3)


def split_and_mix(images1, images2):
    r = (images1[:, :, 0] + images2[:, :, 0]) // 2
    g = (images1[:, :, 1] + images2[:, :, 1]) // 2
    b = (images1[:, :, 2] + images2[:, :, 2]) // 2
    return r, g, b


def output_with_stride_mixed_types(images1, images2):
    return images1[:, :, 2], one_channel_normalize(images2), images1 > images2


def test_split_and_mix():
    run_multi_input_multi_output(split_and_mix)


def test_output_with_stride_mixed_types():
    run_multi_input_multi_output(output_with_stride_mixed_types)


def test_wrong_outputs_number():
    invalid_pipe = TwoOutputsPythonOperatorPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED,
                                                    images_dir, flip)
    invalid_pipe.build()
    try:
        invalid_pipe.run()
    except Exception as e:
        print(e)
        return
    raise Exception('Should not pass')


