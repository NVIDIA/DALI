# Copyright (c) 2019-2024, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import cv2
import glob
import numpy
from nvidia.dali import pipeline_def
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.fn as fn
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import os
import random
import tempfile
import time
from PIL import Image, ImageEnhance
from nvidia.dali.ops import _DataNode
from nose2.tools import params
import numpy as np
from nose_utils import raises
from test_utils import get_dali_extra_path, np_type_to_dali

test_data_root = get_dali_extra_path()
images_dir = os.path.join(test_data_root, "db", "single", "jpeg")


def test_dlpack_conversions():
    array = numpy.arange(0, 10, 0.5)
    reshaped = array.reshape((2, 10, 1))
    slice = reshaped[:, 2:5, :]
    dlpack = ops._dlpack_from_array(slice)
    result_array = ops._dlpack_to_array(dlpack)
    assert result_array.shape == slice.shape
    assert numpy.array_equal(result_array, slice)


def resize(image):
    return numpy.array(Image.fromarray(image).resize((300, 300)))


class CommonPipeline(Pipeline):
    def __init__(
        self, batch_size, num_threads, device_id, _seed, image_dir, prefetch_queue_depth=2
    ):
        super().__init__(
            batch_size,
            num_threads,
            device_id,
            seed=_seed,
            prefetch_queue_depth=prefetch_queue_depth,
        )
        self.input = ops.readers.File(file_root=image_dir)
        self.decode = ops.decoders.Image(device="cpu", output_type=types.RGB)
        self.resize = ops.PythonFunction(function=resize, output_layouts="HWC")

    def load(self):
        jpegs, labels = self.input()
        decoded = self.decode(jpegs)
        resized = self.resize(decoded)
        return resized, labels

    def define_graph(self):
        pass


class BasicPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, seed, image_dir):
        super().__init__(batch_size, num_threads, device_id, seed, image_dir)

    def define_graph(self):
        images, labels = self.load()
        return images


class PythonOperatorPipeline(CommonPipeline):
    def __init__(
        self, batch_size, num_threads, device_id, seed, image_dir, function, prefetch_queue_depth=2
    ):
        super().__init__(
            batch_size,
            num_threads,
            device_id,
            seed,
            image_dir,
            prefetch_queue_depth=prefetch_queue_depth,
        )
        self.python_function = ops.PythonFunction(function=function)

    def define_graph(self):
        images, labels = self.load()
        processed = self.python_function(images)
        assert isinstance(processed, _DataNode)
        return processed


class FlippingPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, seed, image_dir):
        super().__init__(batch_size, num_threads, device_id, seed, image_dir)
        self.flip = ops.Flip(horizontal=1)

    def define_graph(self):
        images, labels = self.load()
        flipped = self.flip(images)
        return flipped


class TwoOutputsPythonOperatorPipeline(CommonPipeline):
    def __init__(
        self, batch_size, num_threads, device_id, seed, image_dir, function, op=ops.PythonFunction
    ):
        super().__init__(batch_size, num_threads, device_id, seed, image_dir)
        self.python_function = op(function=function, num_outputs=2)

    def define_graph(self):
        images, labels = self.load()
        out1, out2 = self.python_function(images)
        assert isinstance(out1, _DataNode)
        assert isinstance(out2, _DataNode)
        return out1, out2


class MultiInputMultiOutputPipeline(CommonPipeline):
    def __init__(
        self, batch_size, num_threads, device_id, seed, image_dir, function, batch_processing=False
    ):
        super().__init__(batch_size, num_threads, device_id, seed, image_dir)
        self.python_function = ops.PythonFunction(
            function=function, num_outputs=3, batch_processing=batch_processing
        )

    def define_graph(self):
        images1, labels1 = self.load()
        images2, labels2 = self.load()
        out1, out2, out3 = self.python_function(images1, images2)
        assert isinstance(out1, _DataNode)
        assert isinstance(out2, _DataNode)
        assert isinstance(out3, _DataNode)
        return out1, out2, out3


class DoubleLoadPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, seed, image_dir):
        super().__init__(batch_size, num_threads, device_id, seed, image_dir)

    def define_graph(self):
        images1, labels1 = self.load()
        images2, labels2 = self.load()
        return images1, images2


class SinkTestPipeline(CommonPipeline):
    def __init__(self, batch_size, device_id, seed, image_dir, function):
        super().__init__(batch_size, 1, device_id, seed, image_dir)
        self.python_function = ops.PythonFunction(function=function, num_outputs=0)

    def define_graph(self):
        images, labels = self.load()
        self.python_function(images)
        return images


class PythonOperatorInputSetsPipeline(PythonOperatorPipeline):
    def __init__(self, batch_size, num_threads, device_id, seed, image_dir, function):
        super().__init__(batch_size, num_threads, device_id, seed, image_dir, function)
        self.python_function = ops.PythonFunction(function=function)

    def define_graph(self):
        images, labels = self.load()
        processed = self.python_function([images, images])
        return processed


def random_seed():
    return int(random.random() * (1 << 32))


DEVICE_ID = 0
BATCH_SIZE = 8
ITERS = 16
SEED = random_seed()
NUM_WORKERS = 6


def run_case(func):
    pipe = BasicPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir)
    pyfunc_pipe = PythonOperatorPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, func)
    for it in range(ITERS):
        (preprocessed_output,) = pipe.run()
        (output,) = pyfunc_pipe.run()
        for i in range(len(output)):
            assert numpy.array_equal(output.at(i), func(preprocessed_output.at(i)))


def one_channel_normalize(image):
    return image[:, :, 1] / 255.0


def channels_mean(image):
    r = numpy.mean(image[:, :, 0])
    g = numpy.mean(image[:, :, 1])
    b = numpy.mean(image[:, :, 2])
    return numpy.array([r, g, b])


def bias(image):
    return numpy.array(image > 127, dtype=bool)


def flip(image):
    return numpy.fliplr(image)


def flip_batch(images):
    return [flip(x) for x in images]


def dlflip(image):
    image = ops._dlpack_to_array(image)
    out = numpy.fliplr(image)
    out = ops._dlpack_from_array(out)
    return out


def dlflip_batch(images):
    return [dlflip(x) for x in images]


def Rotate(image):
    return numpy.rot90(image)


def Brightness(image):
    return numpy.array(ImageEnhance.Brightness(Image.fromarray(image)).enhance(0.5))


def test_python_operator_one_channel_normalize():
    run_case(one_channel_normalize)


def test_python_operator_channels_mean():
    run_case(channels_mean)


def test_python_operator_bias():
    run_case(bias)


def test_python_operator_flip():
    dali_flip = FlippingPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir)
    numpy_flip = PythonOperatorPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, flip)
    for it in range(ITERS):
        (numpy_output,) = numpy_flip.run()
        (dali_output,) = dali_flip.run()
        for i in range(len(numpy_output)):
            assert numpy.array_equal(numpy_output.at(i), dali_output.at(i))


class RotatePipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, seed, image_dir):
        super().__init__(batch_size, num_threads, device_id, seed, image_dir)
        self.rotate = ops.Rotate(angle=90.0, interp_type=types.INTERP_NN)

    def define_graph(self):
        images, labels = self.load()
        rotate = self.rotate(images)
        return rotate


class BrightnessPipeline(CommonPipeline):
    def __init__(self, batch_size, num_threads, device_id, seed, image_dir):
        super().__init__(batch_size, num_threads, device_id, seed, image_dir)
        self.brightness = ops.BrightnessContrast(device="gpu", brightness=0.5)

    def define_graph(self):
        images, labels = self.load()
        bright = self.brightness(images.gpu())
        return bright


def test_python_operator_rotate():
    dali_rotate = RotatePipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir)
    numpy_rotate = PythonOperatorPipeline(
        BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, Rotate
    )
    for it in range(ITERS):
        (numpy_output,) = numpy_rotate.run()
        (dali_output,) = dali_rotate.run()
        for i in range(len(numpy_output)):
            if not numpy.array_equal(numpy_output.at(i), dali_output.at(i)):
                cv2.imwrite("numpy.png", numpy_output.at(i))
                cv2.imwrite("dali.png", dali_output.at(i))
                assert numpy.array_equal(numpy_output.at(i), dali_output.at(i))


def test_python_operator_brightness():
    dali_brightness = BrightnessPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir)
    numpy_brightness = PythonOperatorPipeline(
        BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, Brightness
    )
    for it in range(ITERS):
        (numpy_output,) = numpy_brightness.run()
        (dali_output,) = dali_brightness.run()
        for i in range(len(dali_output)):
            assert numpy.allclose(
                numpy_output.at(i), np.array(dali_output.at(i).as_cpu()), rtol=1e-5, atol=1
            )


def invalid_function(image):
    return img  # noqa: F821. This shall be an invalid function.


@raises(RuntimeError, "img*not defined")
def test_python_operator_invalid_function():
    invalid_pipe = PythonOperatorPipeline(
        BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, invalid_function
    )
    invalid_pipe.run()


@raises(TypeError, "do not support multiple input sets")
def test_python_operator_with_input_sets():
    invalid_pipe = PythonOperatorInputSetsPipeline(
        BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, Rotate
    )
    invalid_pipe.build()


def split_red_blue(image):
    return image[:, :, 0], image[:, :, 2]


def mixed_types(image):
    return bias(image), one_channel_normalize(image)


def run_two_outputs(func):
    pipe = BasicPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir)
    pyfunc_pipe = TwoOutputsPythonOperatorPipeline(
        BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, func
    )
    for it in range(ITERS):
        (preprocessed_output,) = pipe.run()
        output1, output2 = pyfunc_pipe.run()
        for i in range(len(output1)):
            pro1, pro2 = func(preprocessed_output.at(i))
            assert numpy.array_equal(output1.at(i), pro1)
            assert numpy.array_equal(output2.at(i), pro2)


def test_split():
    run_two_outputs(split_red_blue)


def test_mixed_types():
    run_two_outputs(mixed_types)


def multi_per_sample_compare(func, pipe, pyfunc_pipe):
    for it in range(ITERS):
        preprocessed_output1, preprocessed_output2 = pipe.run()
        out1, out2, out3 = pyfunc_pipe.run()
        for i in range(BATCH_SIZE):
            pro1, pro2, pro3 = func(preprocessed_output1.at(i), preprocessed_output2.at(i))
            assert numpy.array_equal(out1.at(i), pro1)
            assert numpy.array_equal(out2.at(i), pro2)
            assert numpy.array_equal(out3.at(i), pro3)


def multi_batch_compare(func, pipe, pyfunc_pipe):
    for it in range(ITERS):
        preprocessed_output1, preprocessed_output2 = pipe.run()
        out1, out2, out3 = pyfunc_pipe.run()
        in1 = [preprocessed_output1.at(i) for i in range(BATCH_SIZE)]
        in2 = [preprocessed_output2.at(i) for i in range(BATCH_SIZE)]
        pro1, pro2, pro3 = func(in1, in2)
        for i in range(BATCH_SIZE):
            assert numpy.array_equal(out1.at(i), pro1[i])
            assert numpy.array_equal(out2.at(i), pro2[i])
            assert numpy.array_equal(out3.at(i), pro3[i])


def run_multi_input_multi_output(func, compare, batch=False):
    pipe = DoubleLoadPipeline(BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir)
    pyfunc_pipe = MultiInputMultiOutputPipeline(
        BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, func, batch_processing=batch
    )
    compare(func, pipe, pyfunc_pipe)


def split_and_mix(images1, images2):
    r = (images1[:, :, 0] + images2[:, :, 0]) // 2
    g = (images1[:, :, 1] + images2[:, :, 1]) // 2
    b = (images1[:, :, 2] + images2[:, :, 2]) // 2
    return r, g, b


def output_with_stride_mixed_types(images1, images2):
    return images1[:, :, 2], one_channel_normalize(images2), images1 > images2


def test_split_and_mix():
    run_multi_input_multi_output(split_and_mix, multi_per_sample_compare)


def test_output_with_stride_mixed_types():
    run_multi_input_multi_output(output_with_stride_mixed_types, multi_per_sample_compare)


def mix_and_split_batch(images1, images2):
    mixed = [(images1[i] + images2[i]) // 2 for i in range(len(images1))]
    r = [im[:, :, 0] for im in mixed]
    g = [im[:, :, 1] for im in mixed]
    b = [im[:, :, 2] for im in mixed]
    return r, g, b


def with_stride_mixed_types_batch(images1, images2):
    out1 = [im[:, :, 2] for im in images1]
    out2 = [one_channel_normalize(im) for im in images2]
    out3 = [im1 > im2 for (im1, im2) in zip(images1, images2)]
    return out1, out2, out3


def test_split_and_mix_batch():
    run_multi_input_multi_output(mix_and_split_batch, multi_batch_compare, batch=True)


def test_output_with_stride_mixed_types_batch():
    run_multi_input_multi_output(with_stride_mixed_types_batch, multi_batch_compare, batch=True)


@raises(Exception, "must be a tuple")
def test_not_a_tuple():
    invalid_pipe = TwoOutputsPythonOperatorPipeline(
        BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, flip_batch
    )
    invalid_pipe.run()


@raises(Exception, "must be a tuple")
def test_not_a_tuple_dl():
    invalid_pipe = TwoOutputsPythonOperatorPipeline(
        BATCH_SIZE,
        NUM_WORKERS,
        DEVICE_ID,
        SEED,
        images_dir,
        dlflip_batch,
        op=ops.DLTensorPythonFunction,
    )
    invalid_pipe.run()


def three_outputs(inp):
    return inp, inp, inp


@raises(Exception, glob="Unexpected number of outputs*got 3*expected 2")
def test_wrong_outputs_number():
    invalid_pipe = TwoOutputsPythonOperatorPipeline(
        BATCH_SIZE, NUM_WORKERS, DEVICE_ID, SEED, images_dir, three_outputs
    )
    invalid_pipe.run()


@raises(Exception, glob="Unexpected number of outputs*got 3*expected 2")
def test_wrong_outputs_number_dl():
    invalid_pipe = TwoOutputsPythonOperatorPipeline(
        BATCH_SIZE,
        NUM_WORKERS,
        DEVICE_ID,
        SEED,
        images_dir,
        three_outputs,
        op=ops.DLTensorPythonFunction,
    )
    invalid_pipe.run()


SINK_PATH = tempfile.mkdtemp()


def save(image):
    Image.fromarray(image).save(SINK_PATH + "/sink_img" + str(time.process_time()) + ".jpg", "JPEG")


def test_sink():
    pipe = SinkTestPipeline(BATCH_SIZE, DEVICE_ID, SEED, images_dir, save)
    if not os.path.exists(SINK_PATH):
        os.mkdir(SINK_PATH)
    assert len(glob.glob(SINK_PATH + "/sink_img*")) == 0
    pipe.run()
    created_files = glob.glob(SINK_PATH + "/sink_img*")
    print(created_files)
    assert len(created_files) == BATCH_SIZE
    for file in created_files:
        os.remove(file)
    os.rmdir(SINK_PATH)


counter = 0


def func_with_side_effects(images):
    global counter
    counter = counter + 1
    return numpy.full_like(images, counter)


def test_func_with_side_effects():
    pipe_one = PythonOperatorPipeline(
        BATCH_SIZE,
        NUM_WORKERS,
        DEVICE_ID,
        SEED,
        images_dir,
        func_with_side_effects,
        prefetch_queue_depth=1,
    )
    pipe_two = PythonOperatorPipeline(
        BATCH_SIZE,
        NUM_WORKERS,
        DEVICE_ID,
        SEED,
        images_dir,
        func_with_side_effects,
        prefetch_queue_depth=1,
    )

    global counter

    for it in range(ITERS):
        counter = 0
        (out_one,) = pipe_one.run()
        (out_two,) = pipe_two.run()
        assert counter == len(out_one) + len(out_two)
        elems_one = [out_one.at(s)[0][0][0] for s in range(BATCH_SIZE)]
        elems_one.sort()
        assert elems_one == [i for i in range(1, BATCH_SIZE + 1)]
        elems_two = [out_two.at(s)[0][0][0] for s in range(BATCH_SIZE)]
        elems_two.sort()
        assert elems_two == [i for i in range(BATCH_SIZE + 1, 2 * BATCH_SIZE + 1)]


class AsyncPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, _seed):
        super().__init__(
            batch_size, num_threads, device_id, seed=_seed, exec_async=True, exec_pipelined=True
        )
        self.op = ops.PythonFunction(function=lambda: numpy.zeros([2, 2, 2]))

    def define_graph(self):
        return self.op()


def test_output_layout():
    pipe = CommonPipeline(1, 1, 0, 999, images_dir)
    with pipe:
        images, _ = pipe.load()
        out1, out2 = fn.python_function(
            images, function=lambda x: (x, x.mean(2)), num_outputs=2, output_layouts=["ABC", "DE"]
        )
        out3, out4 = fn.python_function(
            images, function=lambda x: (x, x / 2), num_outputs=2, output_layouts="FGH"
        )
        out5, out6 = fn.python_function(
            images, function=lambda x: (x, x / 2), num_outputs=2, output_layouts=["IJK"]
        )

        pipe.set_outputs(out1, out2, out3, out4, out5, out6)
    out1, out2, out3, out4, out5, out6 = pipe.run()
    assert out1.layout() == "ABC"
    assert out2.layout() == "DE"
    assert out3.layout() == "FGH"
    assert out4.layout() == "FGH"
    assert out5.layout() == "IJK"
    assert out6.layout() == ""


@raises(RuntimeError, "*length of*output_layouts*greater than*")
def test_invalid_layouts_arg():
    pipe = Pipeline(1, 1, 0, 999, exec_async=False, exec_pipelined=False)
    with pipe:
        out = fn.python_function(function=lambda: numpy.zeros((1, 1)), output_layouts=["HW", "HWC"])
        pipe.set_outputs(out)
    pipe.run()


def test_python_function_conditionals():
    batch_size = 32

    @pipeline_def(
        device_id=0,
        batch_size=batch_size,
        num_threads=4,
        exec_async=False,
        exec_pipelined=False,
        enable_conditionals=True,
    )
    def py_fun_pipeline():
        predicate = fn.external_source(
            source=lambda sample_info: numpy.array(sample_info.idx_in_batch < batch_size / 2),
            batch=False,
        )
        if predicate:
            out1, out2 = fn.python_function(
                predicate, num_outputs=2, function=lambda _: (numpy.array(42), numpy.array(10))
            )
        else:
            out1 = fn.python_function(function=lambda: numpy.array(0))
            out2 = types.Constant(numpy.array(0), device="cpu", dtype=types.INT64)
        return out1, out2

    pipe = py_fun_pipeline()
    pipe.run()


@params(
    numpy.bool_,
    numpy.int_,
    numpy.intc,
    numpy.intp,
    numpy.int8,
    numpy.int16,
    numpy.int32,
    numpy.int64,
    numpy.uint8,
    numpy.uint16,
    numpy.uint32,
    numpy.uint64,
    numpy.float_,
    numpy.float32,
    numpy.float16,
    numpy.short,
    numpy.longlong,
    numpy.ushort,
    numpy.ulonglong,
)
def test_different_types(input_type):
    max_batch_size = 4

    def check_type(data):

        def check_type_fn(data):
            assert data.dtype == input_type

        fn.python_function(data, function=check_type_fn, num_outputs=0, preserve=True)

    @pipeline_def
    def test_pipe():
        data = fn.ones(shape=[1, 1, 1], dtype=np_type_to_dali(input_type))
        check_type(data)
        return data

    pipe = test_pipe(
        batch_size=max_batch_size, num_threads=1, device_id=0, enable_conditionals=True
    )

    _ = pipe.run()


def test_delete_pipe_while_function_running():
    def func(x):
        time.sleep(0.02)
        return x

    for i in range(5):
        with Pipeline(batch_size=1, num_threads=1, device_id=None) as pipe:
            pipe.set_outputs(fn.python_function(types.Constant(0), function=func))
            pipe.run()
        del pipe
