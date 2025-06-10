# Copyright (c) 2020-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import argparse
import nvidia.dali.fn as fn
import nvidia.dali.types as types
import statistics
import time
from nvidia.dali.pipeline import pipeline_def
import random
import numpy as np
import os
from nvidia.dali.auto_aug import auto_augment, trivial_augment


parser = argparse.ArgumentParser(description="DALI HW decoder benchmark")
parser.add_argument("-b", dest="batch_size", help="batch size", default=1, type=int)
parser.add_argument("-m", dest="minibatch_size", help="minibatch size", default=32, type=int)
parser.add_argument("-d", dest="device_id", help="device id", default=0, type=int)
parser.add_argument(
    "-n", dest="gpu_num", help="Number of GPUs used starting from device_id", default=1, type=int
)
parser.add_argument(
    "-g", dest="device", choices=["gpu", "cpu"], help="device to use", default="gpu", type=str
)
parser.add_argument("-w", dest="warmup_iterations", help="warmup iterations", default=0, type=int)
parser.add_argument("-t", dest="total_images", help="total images", default=100, type=int)
parser.add_argument(
    "-j",
    dest="num_threads",
    help="CPU threads. Can be a single value (e.g. 4) or range 'start:end:step' (e.g. 1:8:2). "
    "End of range is included.",
    default="4",
    type=str,
)
parser.add_argument(
    "--exec_dynamic",
    dest="exec_dynamic",
    help="use dynamic excutor",
    default=1,
    type=int,
)
input_files_arg = parser.add_mutually_exclusive_group()
input_files_arg.add_argument("-i", dest="images_dir", help="images dir")
input_files_arg.add_argument(
    "--image_list",
    dest="image_list",
    nargs="+",
    default=[],
    help="List of images used for the benchmark.",
)
parser.add_argument(
    "-p",
    dest="pipeline",
    choices=["decoder", "rn50", "efficientnet_inference", "vit", "efficientnet_training"],
    help="pipeline to test",
    default="decoder",
    type=str,
)
parser.add_argument(
    "--aug-strategy",
    dest="aug_strategy",
    choices=["autoaugment", "trivialaugment", "none"],
    default="autoaugment",
    type=str,
)
parser.add_argument("--width_hint", dest="width_hint", default=0, type=int)
parser.add_argument("--height_hint", dest="height_hint", default=0, type=int)

parser.add_argument(
    "--hw_load",
    dest="hw_load",
    help="HW decoder workload. Can be a single value (e.g. 0.66) or range "
    "'start:end:step' (e.g. 0.0:1.0:0.2). End of range is included.",
    default="0.75",
    type=str,
)


def parse_range_arg(arg_str, parse_fn=int):
    """Parse argument into a list of values. Handles both range format
       'start:end:step' and single values.

    Args:
        arg_str: String argument to parse
        use_float: If True, parse as float values, otherwise as integers
    Returns:
        List of parsed values
    """
    if ":" in arg_str:
        try:
            start, end, step = map(parse_fn, arg_str.split(":"))
            if parse_fn == float:
                return list(np.arange(start, end + step / 2, step))  # +step/2 to include end value
            else:
                return list(range(start, end + 1, step))  # +1 to include end value
        except ValueError:
            raise ValueError(
                f"Invalid range format. Expected 'start:end:step' with {parse_fn.__name__} values"
            )
    else:
        try:
            return [parse_fn(arg_str)]
        except ValueError:
            raise ValueError(f"Invalid value. Expected {parse_fn.__name__} number")


parser.add_argument(
    "--print_every_n_iterations",
    dest="print_every_n_iterations",
    help="If > 0, print statistics every N iterations.",
    default=-1,
    type=int,
)

parser.add_argument(
    "--experimental_decoder",
    action="store_true",
    help="If True, uses the experimental decoder instead of the default",
)

args = parser.parse_args()


@pipeline_def(
    batch_size=args.batch_size,
    num_threads=1,
    device_id=args.device_id,
    seed=0,
    exec_dynamic=args.exec_dynamic,
)
def DecoderPipeline(decoders_module=fn.decoders, hw_load=0):
    device = "mixed" if args.device == "gpu" else "cpu"
    jpegs, _ = fn.readers.file(file_root=args.images_dir)
    images = decoders_module.image(
        jpegs,
        device=device,
        output_type=types.RGB,
        hw_decoder_load=hw_load,
        preallocate_width_hint=args.width_hint,
        preallocate_height_hint=args.height_hint,
    )
    return images


@pipeline_def(
    batch_size=args.batch_size,
    num_threads=1,
    device_id=args.device_id,
    seed=0,
    exec_dynamic=args.exec_dynamic,
)
def RN50Pipeline(minibatch_size, decoders_module=fn.decoders, hw_load=0):
    device = "mixed" if args.device == "gpu" else "cpu"
    jpegs, _ = fn.readers.file(file_root=args.images_dir)
    images = decoders_module.image_random_crop(
        jpegs,
        device=device,
        output_type=types.RGB,
        hw_decoder_load=hw_load,
        preallocate_width_hint=args.width_hint,
        preallocate_height_hint=args.height_hint,
    )
    images = fn.resize(images, resize_x=224, resize_y=224, minibatch_size=minibatch_size)
    layout = types.NCHW
    out_type = types.FLOAT16
    coin_flip = fn.random.coin_flip(probability=0.5)
    images = fn.crop_mirror_normalize(
        images,
        dtype=out_type,
        output_layout=layout,
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        mirror=coin_flip,
    )
    return images


@pipeline_def(
    batch_size=args.batch_size,
    num_threads=1,
    device_id=args.device_id,
    seed=0,
    enable_conditionals=True,
    decoders_module=fn.decoders,
    exec_dynamic=args.exec_dynamic,
)
def EfficientnetTrainingPipeline(
    minibatch_size,
    automatic_augmentation="autoaugment",
    decoders_module=fn.decoders,
    hw_load=0,
):
    dali_device = args.device
    output_layout = types.NCHW
    rng = fn.random.coin_flip(probability=0.5)

    jpegs, _ = fn.readers.file(
        name="Reader",
        file_root=args.images_dir,
    )

    if dali_device == "gpu":
        decoder_device = "mixed"
        resize_device = "gpu"
    else:
        decoder_device = "cpu"
        resize_device = "cpu"

    images = decoders_module.image_random_crop(
        jpegs,
        device=decoder_device,
        output_type=types.RGB,
        random_aspect_ratio=[0.75, 4.0 / 3.0],
        random_area=[0.08, 1.0],
        hw_decoder_load=hw_load,
        preallocate_width_hint=args.width_hint,
        preallocate_height_hint=args.height_hint,
    )

    images = fn.resize(
        images,
        device=resize_device,
        size=[224, 224],
        antialias=False,
        minibatch_size=minibatch_size,
    )

    # Make sure that from this point we are processing on GPU regardless
    # of dali_device parameter
    images = images.gpu()

    images = fn.flip(images, horizontal=rng)

    # Based on the specification, apply the automatic augmentation policy. Note, that from
    # the pointof Pipeline definition, this `if` statement relies on static scalar
    # parameter, so it is evaluated exactly once during build - we either include automatic
    # augmentations or not.We pass the shape of the image after the resize so
    # the translate operations are done relative to the image size.
    if automatic_augmentation == "autoaugment":
        output = auto_augment.auto_augment_image_net(images, shape=[224, 224])
    elif automatic_augmentation == "trivialaugment":
        output = trivial_augment.trivial_augment_wide(images, shape=[224, 224])
    else:
        output = images

    output = fn.crop_mirror_normalize(
        output,
        dtype=types.FLOAT,
        output_layout=output_layout,
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )

    return output


@pipeline_def(
    batch_size=args.batch_size,
    num_threads=1,
    device_id=args.device_id,
    prefetch_queue_depth=1,
    exec_dynamic=args.exec_dynamic,
)
def EfficientnetInferencePipeline(decoders_module=fn.decoders, hw_load=0):
    images = fn.external_source(device="cpu", name=DALI_INPUT_NAME)
    images = decoders_module.image(
        images,
        device="mixed" if args.device == "gpu" else "cpu",
        output_type=types.RGB,
        hw_decoder_load=hw_load,
    )
    images = fn.resize(images, resize_x=224, resize_y=224)
    images = fn.crop_mirror_normalize(
        images,
        dtype=types.FLOAT,
        output_layout="CHW",
        crop=(224, 224),
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
    )
    return images


def feed_input(dali_pipeline, data):
    if needs_feed_input:
        assert data is not None, "Input data has not been provided."
        dali_pipeline.feed_input(DALI_INPUT_NAME, data)


def create_input_tensor(batch_size, file_list):
    """
    Creates an input batch to the DALI Pipeline.
    The batch will comprise the files defined within file list and will be shuffled.
    If the file list contains fewer files than the batch size, they will be repeated.
    The encoded images will be padded.
    :param batch_size: Requested batch size.
    :param file_list: List of images to be loaded.
    :return:
    """
    # Adjust file_list to batch_size
    while len(file_list) < batch_size:
        file_list += file_list
    file_list = file_list[:batch_size]

    random.shuffle(file_list)

    # Read the files as byte buffers
    arrays = list(map(lambda x: np.fromfile(x, dtype=np.uint8), file_list))

    # Pad the encoded images
    lengths = list(map(lambda x, ar=arrays: ar[x].shape[0], [x for x in range(len(arrays))]))
    max_len = max(lengths)
    arrays = list(map(lambda ar, ml=max_len: np.pad(ar, (0, ml - ar.shape[0])), arrays))

    for arr in arrays:
        assert arr.shape == arrays[0].shape, "Arrays must have the same shape"
    return np.stack(arrays)


def non_image_preprocessing(raw_text):
    return np.array([int(bytes(raw_text).decode("utf-8"))])


@pipeline_def(batch_size=args.batch_size, num_threads=1, device_id=args.device_id, seed=0)
def vit_pipeline(
    is_training=False,
    image_shape=(384, 384, 3),
    num_classes=1000,
    decoders_module=fn.decoders,
    hw_load=0,
    exec_dynamic=args.exec_dynamic,
):
    files_paths = [os.path.join(args.images_dir, f) for f in os.listdir(args.images_dir)]

    img, clss = fn.readers.webdataset(
        paths=files_paths,
        index_paths=None,
        ext=["jpg", "cls"],
        missing_component_behavior="error",
        random_shuffle=False,
        shard_id=0,
        num_shards=1,
        pad_last_batch=False if is_training else True,
        name="webdataset_reader",
    )

    use_gpu = args.device == "gpu"
    labels = fn.python_function(clss, function=non_image_preprocessing, num_outputs=1)
    if use_gpu:
        labels = labels.gpu()
    labels = fn.one_hot(labels, num_classes=num_classes)

    device = "mixed" if use_gpu else "cpu"
    img = decoders_module.image(
        img,
        device=device,
        output_type=types.RGB,
        hw_decoder_load=hw_load,
        preallocate_width_hint=args.width_hint,
        preallocate_height_hint=args.height_hint,
    )

    if is_training:
        img = fn.random_resized_crop(img, size=image_shape[:-1])
        img = fn.flip(img, depthwise=0, horizontal=fn.random.coin_flip())

        # color jitter
        brightness = fn.random.uniform(range=[0.6, 1.4])
        contrast = fn.random.uniform(range=[0.6, 1.4])
        saturation = fn.random.uniform(range=[0.6, 1.4])
        hue = fn.random.uniform(range=[0.9, 1.1])
        img = fn.color_twist(
            img, brightness=brightness, contrast=contrast, hue=hue, saturation=saturation
        )

        # auto-augment
        # `shape` controls the magnitude of the translation operations
        img = auto_augment.auto_augment_image_net(img)
    else:
        img = fn.resize(img, size=image_shape[:-1])

    # normalize
    # https://github.com/NVIDIA/DALI/issues/4469
    mean = np.asarray([0.485, 0.456, 0.406])[None, None, :]
    std = np.asarray([0.229, 0.224, 0.225])[None, None, :]
    scale = 1 / 255.0
    img = fn.normalize(img, mean=mean / scale, stddev=std, scale=scale, dtype=types.FLOAT)

    return img, labels


DALI_INPUT_NAME = "DALI_INPUT_0"
needs_feed_input = args.pipeline == "efficientnet_inference"

threads_num = parse_range_arg(args.num_threads, parse_fn=int)
decoder_hw_load = parse_range_arg(args.hw_load, parse_fn=float)

print(f"Threads num to check: {threads_num}")
print(f"Decoder hw load to check: {decoder_hw_load}")

perf_results = []
for cpu_num in threads_num:
    for hw_load in decoder_hw_load:
        decoders_module = fn.experimental.decoders if args.experimental_decoder else fn.decoders
        print(f"Using decoders_module={decoders_module}")

        pipes = []
        if args.pipeline == "decoder":
            for i in range(args.gpu_num):
                pipes.append(
                    DecoderPipeline(
                        device_id=i + args.device_id,
                        num_threads=cpu_num,
                        decoders_module=decoders_module,
                        hw_load=hw_load,
                    )
                )
        elif args.pipeline == "rn50":
            for i in range(args.gpu_num):
                pipes.append(
                    RN50Pipeline(
                        device_id=i + args.device_id,
                        minibatch_size=args.minibatch_size,
                        num_threads=cpu_num,
                        decoders_module=decoders_module,
                        hw_load=hw_load,
                    )
                )
        elif args.pipeline == "efficientnet_inference":
            for i in range(args.gpu_num):
                pipes.append(
                    EfficientnetInferencePipeline(
                        device_id=i + args.device_id,
                        num_threads=cpu_num,
                        decoders_module=decoders_module,
                        hw_load=hw_load,
                    )
                )
        elif args.pipeline == "vit":
            for i in range(args.gpu_num):
                pipes.append(
                    vit_pipeline(
                        device_id=i + args.device_id,
                        num_threads=cpu_num,
                        decoders_module=decoders_module,
                        hw_load=hw_load,
                    )
                )
        elif args.pipeline == "efficientnet_training":
            for i in range(args.gpu_num):
                pipes.append(
                    EfficientnetTrainingPipeline(
                        device_id=i + args.device_id,
                        minibatch_size=args.minibatch_size,
                        automatic_augmentation=args.aug_strategy,
                        num_threads=cpu_num,
                        decoders_module=decoders_module,
                        hw_load=hw_load,
                    )
                )
        else:
            raise RuntimeError("Unsupported pipeline")
        for p in pipes:
            p.build()

        input_tensor = (
            create_input_tensor(args.batch_size, args.image_list) if needs_feed_input else None
        )

        for iteration in range(args.warmup_iterations):
            for p in pipes:
                feed_input(p, input_tensor)
                p.schedule_run()
            for p in pipes:
                _ = p.share_outputs()
            for p in pipes:
                p.release_outputs()
        print("Warmup finished")

        test_iterations = args.total_images // args.batch_size

        print("Test iterations: ", test_iterations)
        start_time = time.perf_counter()
        execution_times = []
        for iteration in range(test_iterations):
            iter_start_time = time.perf_counter()
            for p in pipes:
                feed_input(p, input_tensor)
                p.schedule_run()
            for p in pipes:
                _ = p.share_outputs()
            for p in pipes:
                p.release_outputs()
            iter_end_time = time.perf_counter()
            iter_duration = iter_end_time - iter_start_time
            execution_times.append(iter_duration)

            if args.print_every_n_iterations > 0 and (
                (iteration + 1) % args.print_every_n_iterations == 0
                or iteration == test_iterations - 1
            ):
                elapsed_time = time.perf_counter() - start_time
                throughput = (iteration + 1) * args.batch_size * args.gpu_num / elapsed_time
                mean_t = statistics.mean(execution_times)
                median_t = statistics.median(execution_times)
                min_t = min(execution_times)
                max_t = max(execution_times)
                print(
                    f"Iteration {iteration + 1}/{test_iterations} - "
                    + f"Throughput: {throughput:.2f} frames/sec "
                    + f"(mean={mean_t:.6f}sec, median={median_t:.6f}sec, "
                    + f"min={min_t:.6f}sec, max={max_t:.6f}sec)"
                )

        end_time = time.perf_counter()
        total_time = end_time - start_time
        total_throughput = test_iterations * args.batch_size * args.gpu_num / total_time
        avg_t = statistics.mean(execution_times)
        stdev_t = statistics.stdev(execution_times)
        median_t = statistics.median(execution_times)
        min_t = min(execution_times)
        max_t = max(execution_times)

        print("\nFinal Results:")
        print(f"CPU threads: {cpu_num}")
        print(f"HW decoder load: {hw_load}")
        print(f"Total Execution Time: {total_time:.6f} sec")
        print(f"Total Throughput: {total_throughput:.2f} frames/sec")
        print(f"Average time per iteration: {avg_t:.6f} sec")
        print(f"Median time per iteration: {median_t:.6f} sec")
        print(f"Stddev time per iteration: {stdev_t:.6f} sec")
        print(f"Min time per iteration: {min_t:.6f} sec")
        print(f"Max time per iteration: {max_t:.6f} sec")
        perf_results.append(
            {
                "cpu_num": cpu_num,
                "hw_load": hw_load,
                "total_time": total_time,
                "total_throughput": total_throughput,
            }
        )

if len(perf_results) > 0:
    # Find and print result with best throughputif
    best_result = max(perf_results, key=lambda x: x["total_throughput"])
    print("\nBest throughput configuration:")
    print(f"Best: CPU threads: {best_result['cpu_num']}")
    print(f"Best: HW decoder load: {best_result['hw_load']}")
    print(f"Best: Total time: {best_result['total_time']:.6f} sec")
    print(f"Best: Throughput: {best_result['total_throughput']:.2f} frames/sec")
else:
    print("No results to display")
