# Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import argparse
import time
from test_utils import get_dali_extra_path
import os

data_paths = os.path.join(get_dali_extra_path(), "db", "single", "jpeg")


class RN50Pipeline(Pipeline):
    def __init__(
        self, batch_size, num_threads, device_id, num_gpus, data_paths, prefetch, fp16, nhwc
    ):
        super(RN50Pipeline, self).__init__(
            batch_size, num_threads, device_id, prefetch_queue_depth=prefetch
        )
        self.input = ops.readers.File(file_root=data_paths, shard_id=device_id, num_shards=num_gpus)
        self.decode_gpu = ops.decoders.Image(device="mixed", output_type=types.RGB)
        self.res = ops.RandomResizedCrop(device="gpu", size=(224, 224))

        layout = types.args.nhwc if nhwc else types.NCHW
        out_type = types.FLOAT16 if fp16 else types.FLOAT

        self.cmnp = ops.CropMirrorNormalize(
            device="gpu",
            dtype=out_type,
            output_layout=layout,
            crop=(224, 224),
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        self.coin = ops.random.CoinFlip(probability=0.5)

    def define_graph(self):
        rng = self.coin()
        jpegs, labels = self.input(name="Reader")
        images = self.decode_gpu(jpegs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return (output, labels.gpu())


parser = argparse.ArgumentParser(
    description="Test RN50 augmentation pipeline with different FW iterators"
)
parser.add_argument(
    "-g", "--gpus", default=1, type=int, metavar="N", help="number of GPUs (default: 1)"
)
parser.add_argument(
    "-b", "--batch_size", default=13, type=int, metavar="N", help="batch size (default: 13)"
)
parser.add_argument(
    "-p", "--print-freq", default=10, type=int, metavar="N", help="print frequency (default: 10)"
)
parser.add_argument(
    "-j",
    "--workers",
    default=3,
    type=int,
    metavar="N",
    help="number of data loading workers (default: 3)",
)
parser.add_argument(
    "--prefetch", default=2, type=int, metavar="N", help="prefetch queue depth (default: 2)"
)
parser.add_argument("--separate_queue", action="store_true", help="Use separate queues executor")
parser.add_argument(
    "--cpu_size", default=2, type=int, metavar="N", help="cpu prefetch queue depth (default: 2)"
)
parser.add_argument(
    "--gpu_size", default=2, type=int, metavar="N", help="gpu prefetch queue depth (default: 2)"
)
parser.add_argument("--fp16", action="store_true", help="Run fp16 pipeline")
parser.add_argument(
    "--nhwc", action="store_true", help="Use args.nhwc data instead of default NCHW"
)
parser.add_argument(
    "-i",
    "--iters",
    default=-1,
    type=int,
    metavar="N",
    help="Number of iterations to run (default: -1 - whole data set)",
)
parser.add_argument(
    "-e", "--epochs", default=1, type=int, metavar="N", help="Number of epochs to run (default: 1)"
)
parser.add_argument("--framework", type=str)
args = parser.parse_args()

print(
    f"Framework: {args.framework}, GPUs: {args.gpus}, batch: {args.batch_size}, "
    f"workers: {args.workers}, prefetch depth: {args.prefetch}, "
    f"loging interval: {args.print_freq}, fp16: {args.fp16}, args.nhwc: {args.nhwc}"
)


PREFETCH = args.prefetch
if args.separate_queue:
    PREFETCH = {"cpu_size": args.cpu_size, "gpu_size": args.gpu_size}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.avg_last_n = 0
        self.max_val = 0

    def update(self, val, n=1):
        self.val = val
        self.max_val = max(self.max_val, val)
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def test_fw_iter(IteratorClass, args):
    iterator_name = IteratorClass.__module__ + "." + IteratorClass.__name__
    print("Start testing {}".format(iterator_name))
    sess = None
    daliop = None
    dali_train_iter = None
    images = []
    labels = []

    pipes = [
        RN50Pipeline(
            batch_size=args.batch_size,
            num_threads=args.workers,
            device_id=n,
            num_gpus=args.gpus,
            data_paths=data_paths,
            prefetch=PREFETCH,
            fp16=args.fp16,
            nhwc=args.nhwc,
        )
        for n in range(args.gpus)
    ]
    [pipe.build() for pipe in pipes]
    iters = args.iters
    if args.iters < 0:
        iters = pipes[0].epoch_size("Reader")
        assert all(pipe.epoch_size("Reader") == iters for pipe in pipes)
        iters_tmp = iters
        iters = iters // args.batch_size
        if iters_tmp != iters * args.batch_size:
            iters += 1
        iters_tmp = iters

        iters = iters // args.gpus
        if iters_tmp != iters * args.gpus:
            iters += 1

    if iterator_name == "nvidia.dali.plugin.tf.DALIIterator":
        daliop = IteratorClass()
        for dev in range(args.gpus):
            with tf.device("/gpu:%i" % dev):
                if args.fp16:
                    out_type = tf.float16
                else:
                    out_type = tf.float32
                image, label = daliop(
                    pipeline=pipes[dev],
                    shapes=[(args.batch_size, 3, 224, 224), ()],
                    dtypes=[out_type, tf.int32],
                )
                images.append(image)
                labels.append(label)
        gpu_options = GPUOptions(per_process_gpu_memory_fraction=0.5)
        config = ConfigProto(gpu_options=gpu_options)
        sess = Session(config=config)

    end = time.time()
    for i in range(args.epochs):
        if i == 0:
            print("Warm up")
        else:
            print("Test run " + str(i))
        data_time = AverageMeter()

        if iterator_name == "nvidia.dali.plugin.tf.DALIIterator":
            assert sess is not None
            for j in range(iters):
                sess.run([images, labels])
                data_time.update(time.time() - end)
                if j % args.print_freq == 0:
                    speed = args.gpus * args.batch_size / data_time.avg
                    print(
                        f"{iterator_name} {j + 1}/ {iters}, avg time: {data_time.avg} [s], "
                        f"worst time: {data_time.max_val} [s], speed: {speed} [img/s]"
                    )
                end = time.time()
        else:
            dali_train_iter = IteratorClass(pipes, reader_name="Reader")
            j = 0
            for it in iter(dali_train_iter):
                data_time.update(time.time() - end)
                if j % args.print_freq == 0:
                    speed = args.gpus * args.batch_size / data_time.avg
                    print(
                        f"{iterator_name} {j + 1}/ {iters}, avg time: {data_time.avg} [s], "
                        f"worst time: {data_time.max_val} [s], speed: {speed} [img/s]"
                    )
                end = time.time()
                j = j + 1
                if j > iters:
                    break


def import_mxnet():
    from nvidia.dali.plugin.mxnet import DALIClassificationIterator as MXNetIterator

    return MXNetIterator


def import_pytorch():
    from nvidia.dali.plugin.pytorch import DALIClassificationIterator as PyTorchIterator

    return PyTorchIterator


def import_paddle():
    from nvidia.dali.plugin.paddle import DALIClassificationIterator as PaddleIterator

    return PaddleIterator


def import_tf():
    global tf
    global GPUOptions
    global ConfigProto
    global Session
    from nvidia.dali.plugin.tf import DALIIterator as TensorFlowIterator
    import tensorflow as tf

    try:
        from tensorflow.compat.v1 import GPUOptions
        from tensorflow.compat.v1 import ConfigProto
        from tensorflow.compat.v1 import Session
    except ImportError:
        # Older TF versions don't have compat.v1 layer
        from tensorflow import GPUOptions
        from tensorflow import ConfigProto
        from tensorflow import Session

    try:
        tf.compat.v1.disable_eager_execution()
    except NameError:
        pass
    return TensorFlowIterator


Iterators = {
    "mxnet": [import_mxnet],
    "pytorch": [import_pytorch],
    "tf": [import_tf],
    "paddle": [import_paddle],
}

assert args.framework in Iterators, "Error, framework {} not supported".format(args.framework)
for imports in Iterators[args.framework]:
    IteratorClass = imports()
    test_fw_iter(IteratorClass, args)
