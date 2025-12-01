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

from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
import nvidia.dali.types as types
import nvidia.dali.tfrecord as tfrec
import os
import glob
import argparse
import time
from test_utils import get_dali_extra_path, AverageMeter
from subprocess import call
import tempfile


class CommonPipeline(Pipeline):
    def __init__(
        self,
        data_paths,
        num_shards,
        batch_size,
        num_threads,
        device_id,
        prefetch,
        fp16,
        random_shuffle,
        nhwc,
        dont_use_mmap,
        decoder_type,
        decoder_cache_params,
        reader_queue_depth,
        shard_id,
    ):
        super(CommonPipeline, self).__init__(
            batch_size, num_threads, device_id, random_shuffle, prefetch_queue_depth=prefetch
        )
        print(f"decoder type: {decoder_type}")
        if "experimental" in decoder_type:
            decoders_module = ops.experimental.decoders
        else:
            decoders_module = ops.decoders

        if "roi" in decoder_type:
            print("Using nvJPEG with ROI decoding")
            self.decode_gpu = decoders_module.ImageRandomCrop(device="mixed", output_type=types.RGB)
            self.res = ops.Resize(device="gpu", resize_x=224, resize_y=224)
        elif "cached" in decoder_type:
            assert decoder_cache_params["cache_enabled"]
            cache_size = decoder_cache_params["cache_size"]
            cache_threshold = decoder_cache_params["cache_threshold"]
            cache_type = decoder_cache_params["cache_type"]
            print(
                f"Using nvJPEG with cache (size : {cache_size} "
                f"threshold: {cache_threshold}, type: {cache_type})"
            )
            self.decode_gpu = decoders_module.Image(
                device="mixed",
                output_type=types.RGB,
                cache_size=cache_size,
                cache_threshold=cache_threshold,
                cache_type=cache_type,
                cache_debug=False,
            )
            self.res = ops.RandomResizedCrop(device="gpu", size=(224, 224))
        else:
            print("Using nvJPEG")
            self.decode_gpu = decoders_module.Image(device="mixed", output_type=types.RGB)
            self.res = ops.RandomResizedCrop(device="gpu", size=(224, 224))

        layout = types.NHWC if nhwc else types.NCHW
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

    def base_define_graph(self, inputs, labels):
        rng = self.coin()
        images = self.decode_gpu(inputs)
        images = self.res(images)
        output = self.cmnp(images.gpu(), mirror=rng)
        return (output, labels)


class MXNetReaderPipeline(CommonPipeline):
    def __init__(self, **kwargs):
        super(MXNetReaderPipeline, self).__init__(**kwargs)
        cache_enabled = kwargs["decoder_cache_params"]["cache_enabled"]
        self.input = ops.readers.MXNet(
            path=kwargs["data_paths"][0],
            index_path=kwargs["data_paths"][1],
            shard_id=kwargs["shard_id"],
            num_shards=kwargs["num_shards"],
            random_shuffle=kwargs["random_shuffle"],
            stick_to_shard=cache_enabled,
            prefetch_queue_depth=kwargs["reader_queue_depth"],
        )

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)


class CaffeReadPipeline(CommonPipeline):
    def __init__(self, **kwargs):
        super(CaffeReadPipeline, self).__init__(**kwargs)
        cache_enabled = kwargs["decoder_cache_params"]["cache_enabled"]
        self.input = ops.readers.Caffe(
            path=kwargs["data_paths"][0],
            shard_id=kwargs["shard_id"],
            num_shards=kwargs["num_shards"],
            stick_to_shard=cache_enabled,
            random_shuffle=kwargs["random_shuffle"],
            dont_use_mmap=kwargs["dont_use_mmap"],
            prefetch_queue_depth=kwargs["reader_queue_depth"],
        )

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)


class Caffe2ReadPipeline(CommonPipeline):
    def __init__(self, **kwargs):
        super(Caffe2ReadPipeline, self).__init__(**kwargs)
        cache_enabled = kwargs["decoder_cache_params"]["cache_enabled"]
        self.input = ops.readers.Caffe2(
            path=kwargs["data_paths"][0],
            shard_id=kwargs["shard_id"],
            num_shards=kwargs["num_shards"],
            random_shuffle=kwargs["random_shuffle"],
            dont_use_mmap=kwargs["dont_use_mmap"],
            stick_to_shard=cache_enabled,
            prefetch_queue_depth=kwargs["reader_queue_depth"],
        )

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)


class FileReadPipeline(CommonPipeline):
    def __init__(self, **kwargs):
        super(FileReadPipeline, self).__init__(**kwargs)
        cache_enabled = kwargs["decoder_cache_params"]["cache_enabled"]
        self.input = ops.readers.File(
            file_root=kwargs["data_paths"][0],
            shard_id=kwargs["shard_id"],
            num_shards=kwargs["num_shards"],
            random_shuffle=kwargs["random_shuffle"],
            dont_use_mmap=kwargs["dont_use_mmap"],
            stick_to_shard=cache_enabled,
            prefetch_queue_depth=kwargs["reader_queue_depth"],
        )

    def define_graph(self):
        images, labels = self.input(name="Reader")
        return self.base_define_graph(images, labels)


class TFRecordPipeline(CommonPipeline):
    def __init__(self, **kwargs):
        super(TFRecordPipeline, self).__init__(**kwargs)
        tfrecord = sorted(glob.glob(kwargs["data_paths"][0]))
        tfrecord_idx = sorted(glob.glob(kwargs["data_paths"][1]))
        if len(tfrecord_idx) == 0:
            # generate indices
            self.temp_dir = tempfile.TemporaryDirectory()
            tfrecord_idxs = [
                os.path.join(self.temp_dir.name, f"{os.path.basename(f)}.idx") for f in tfrecord
            ]
            for tfrecord_file, tfrecord_idx_file in zip(tfrecord, tfrecord_idxs):
                print(f"Generating index file for {tfrecord_file}")
                call(["tfrecord2idx", tfrecord_file, tfrecord_idx_file])
            tfrecord_idx = tfrecord_idxs
        cache_enabled = kwargs["decoder_cache_params"]["cache_enabled"]
        self.input = ops.readers.TFRecord(
            path=tfrecord,
            index_path=tfrecord_idx,
            shard_id=kwargs["shard_id"],
            num_shards=kwargs["num_shards"],
            random_shuffle=kwargs["random_shuffle"],
            dont_use_mmap=kwargs["dont_use_mmap"],
            stick_to_shard=cache_enabled,
            features={
                "image/encoded": tfrec.FixedLenFeature((), tfrec.string, ""),
                "image/class/label": tfrec.FixedLenFeature([1], tfrec.int64, -1),
            },
        )

    def define_graph(self):
        inputs = self.input(name="Reader")
        images = inputs["image/encoded"]
        labels = inputs["image/class/label"]
        return self.base_define_graph(images, labels)


class WebdatasetPipeline(CommonPipeline):
    def __init__(
        self,
        data_paths,
        decoder_cache_params,
        shard_id,
        num_shards,
        random_shuffle,
        dont_use_mmap,
        **kwargs,
    ):
        super(WebdatasetPipeline, self).__init__(
            data_paths=data_paths,
            decoder_cache_params=decoder_cache_params,
            shard_id=shard_id,
            num_shards=num_shards,
            random_shuffle=random_shuffle,
            dont_use_mmap=dont_use_mmap,
            **kwargs,
        )
        wds, wds_idx = data_paths[:2]

        cache_enabled = decoder_cache_params["cache_enabled"]
        self.input = ops.readers.Webdataset(
            paths=wds,
            index_paths=wds_idx,
            ext=["jpg", "cls"],
            shard_id=shard_id,
            num_shards=num_shards,
            random_shuffle=random_shuffle,
            dont_use_mmap=dont_use_mmap,
            stick_to_shard=cache_enabled,
        )

    def define_graph(self):
        return self.base_define_graph(*self.input(name="Reader"))


test_data = {
    # RecordIO & LMDB are not that frequently used any more so we won't test full datasets,
    # just a small ones
    FileReadPipeline: [["/data/imagenet/train-jpeg"], ["/data/imagenet/val-jpeg"]],
    TFRecordPipeline: [
        [
            "/data/imagenet/train-val-tfrecord/train-*",
            "/data/imagenet/train-val-tfrecord.idx/train-*",
        ]
    ],
}

data_root = get_dali_extra_path()

small_test_data = {
    FileReadPipeline: [[os.path.join(data_root, "db/single/jpeg/")]],
    MXNetReaderPipeline: [
        [
            os.path.join(data_root, "db/recordio/train.rec"),
            os.path.join(data_root, "db/recordio/train.idx"),
        ]
    ],
    CaffeReadPipeline: [[os.path.join(data_root, "db/lmdb")]],
    Caffe2ReadPipeline: [[os.path.join(data_root, "db/c2lmdb")]],
    TFRecordPipeline: [
        [
            os.path.join(data_root, "db/tfrecord/train"),
            os.path.join(data_root, "db/tfrecord/train.idx"),
        ]
    ],
    WebdatasetPipeline: [
        [
            os.path.join(data_root, "db/webdataset/train.tar"),
            os.path.join(data_root, "db/webdataset/train.idx"),
        ]
    ],
}

parser = argparse.ArgumentParser(
    description="Test nvJPEG based RN50 augmentation pipeline with different datasets"
)
parser.add_argument(
    "-g",
    "--gpus",
    default=1,
    type=int,
    metavar="N",
    help="number of GPUs run in parallel by this test (default: 1)",
)
parser.add_argument(
    "-b", "--batch", default=512, type=int, metavar="N", help="batch size (default: 512)"
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
parser.add_argument("--nhwc", action="store_true", help="Use NHWC data instead of default NCHW")
parser.add_argument(
    "-i",
    "--iters",
    default=-1,
    type=int,
    metavar="N",
    help="Number of iterations to run (default: -1 - whole data set)",
)
parser.add_argument("--epochs", default=2, type=int, metavar="N", help="Number of epochs to run")
parser.add_argument(
    "--decoder_type",
    default="",
    type=str,
    metavar="N",
    help="roi, cached, (default: regular nvjpeg). Also admit +experimental",
)
parser.add_argument("--cache_size", default=0, type=int, metavar="N", help="Cache size (in MB)")
parser.add_argument("--cache_threshold", default=0, type=int, metavar="N", help="Cache threshold")
parser.add_argument("--cache_type", default="none", type=str, metavar="N", help="Cache type")
parser.add_argument(
    "--reader_queue_depth",
    default=1,
    type=int,
    metavar="N",
    help="prefetch queue depth (default: 1)",
)
parser.add_argument("--read_shuffle", action="store_true", help="Shuffle data when reading")
parser.add_argument(
    "--disable_mmap",
    action="store_true",
    help="Disable mmap for DALI readers. Used for network filesystem tests.",
)
parser.add_argument(
    "-s", "--small", action="store_true", help="use small dataset, DALI_EXTRA_PATH needs to be set"
)
parser.add_argument(
    "--number_of_shards",
    default=None,
    type=int,
    metavar="N",
    help="Number of shards in the dataset",
)
parser.add_argument(
    "--assign_gpu",
    default=None,
    type=int,
    metavar="N",
    help="Assign a given GPU. Cannot be used with --gpus",
)
parser.add_argument(
    "--assign_shard",
    default=0,
    type=int,
    metavar="N",
    help="Assign a given shard id. If used with --gpus, it assigns "
    "the first GPU to this id and next GPUs get consecutive ids",
)
parser.add_argument(
    "--simulate_N_gpus",
    default=None,
    type=int,
    metavar="N",
    help="Used to simulate small shard as it would be in a multi gpu setup "
    "with this number of gpus. If provided, each gpu will see a shard "
    "size as if we were in a multi gpu setup with this number of gpus",
    dest="number_of_shards",
)
parser.add_argument(
    "--remove_default_pipeline_paths",
    action="store_true",
    help="For all data pipeline types, remove the default values",
)
parser.add_argument(
    "--file_read_pipeline_paths",
    default=None,
    type=str,
    metavar="N",
    help="Add custom FileReadPipeline paths. Separate multiple paths by commas",
)
parser.add_argument(
    "--mxnet_reader_pipeline_paths",
    default=None,
    type=str,
    metavar="N",
    help="Add custom MXNetReaderPipeline paths. For a given path, a .rec and .idx "
    "extension will be appended. Separate multiple paths by commas",
)
parser.add_argument(
    "--caffe_read_pipeline_paths",
    default=None,
    type=str,
    metavar="N",
    help="Add custom CaffeReadPipeline paths. Separate multiple paths by commas",
)
parser.add_argument(
    "--caffe2_read_pipeline_paths",
    default=None,
    type=str,
    metavar="N",
    help="Add custom Caffe2ReadPipeline paths. Separate multiple paths by commas",
)
parser.add_argument(
    "--tfrecord_pipeline_paths",
    default=None,
    type=str,
    metavar="N",
    help="Add custom TFRecordPipeline paths. For a given path, a second path with "
    "an .idx extension will be added for the required idx file(s). Separate "
    "multiple paths by commas",
)
parser.add_argument(
    "--webdataset_pipeline_paths",
    default=None,
    type=str,
    metavar="N",
    help="Add custom WebdatasetPipeline paths. For a given path, a second path "
    "with an .idx extension will be added for the required idx file(s). "
    "Separate multiple paths by commas",
)
parser.add_argument(
    "--system_id",
    default="localhost",
    type=str,
    metavar="N",
    help="Add a system id to denote a unique identifier for the performance output. "
    "Defaults to localhost",
)
args = parser.parse_args()

N = args.gpus  # number of GPUs
GPU_ID = args.assign_gpu
DALI_SHARD = args.assign_shard
BATCH_SIZE = args.batch  # batch size
LOG_INTERVAL = args.print_freq
WORKERS = args.workers
PREFETCH = args.prefetch
if args.separate_queue:
    PREFETCH = {"cpu_size": args.cpu_size, "gpu_size": args.gpu_size}
FP16 = args.fp16
NHWC = args.nhwc
SYS_ID = args.system_id

if args.remove_default_pipeline_paths:
    for pipe_name in test_data.keys():
        test_data[pipe_name] = []

if args.file_read_pipeline_paths:
    paths = args.file_read_pipeline_paths.split(",")
    for path in paths:
        test_data[FileReadPipeline].append([path])

if args.mxnet_reader_pipeline_paths:
    paths = args.mxnet_reader_pipeline_paths.split(",")
    for path in paths:
        path_expanded = [path + ".rec", path + ".idx"]
        test_data[MXNetReaderPipeline].append(path_expanded)

if args.caffe_read_pipeline_paths:
    paths = args.caffe_read_pipeline_paths.split(",")
    for path in paths:
        test_data[CaffeReadPipeline].append([path])

if args.caffe2_read_pipeline_paths:
    paths = args.caffe2_read_pipeline_paths.split(",")
    for path in paths:
        test_data[Caffe2ReadPipeline].append([path])

if args.tfrecord_pipeline_paths:
    paths = args.tfrecord_pipeline_paths.split(",")
    for path in paths:
        idx_split_path, idx_split_file = os.path.split(path)
        idx_split_path = idx_split_path + ".idx"
        idx_path = os.path.join(idx_split_path, idx_split_file)
        path_expanded = [path, idx_path]
        test_data[TFRecordPipeline].append(path_expanded)
if args.webdataset_pipeline_paths:
    paths = args.webdataset_pipeline_paths.split(",")
    for path in paths:
        idx_split_path, idx_split_file = os.path.split(path)
        idx_split_path = idx_split_path + ".idx"
        idx_path = os.path.join(idx_split_path, idx_split_file)
        path_expanded = [path, idx_path]
        test_data[WebdatasetPipeline].append(path_expanded)


DECODER_TYPE = args.decoder_type
CACHED_DECODING = DECODER_TYPE == "cached"
DECODER_CACHE_PARAMS = {}
DECODER_CACHE_PARAMS["cache_enabled"] = CACHED_DECODING
if CACHED_DECODING:
    DECODER_CACHE_PARAMS["cache_type"] = args.cache_type
    DECODER_CACHE_PARAMS["cache_size"] = args.cache_size
    DECODER_CACHE_PARAMS["cache_threshold"] = args.cache_threshold
READER_QUEUE_DEPTH = args.reader_queue_depth
NUMBER_OF_SHARDS = N if args.number_of_shards is None else args.number_of_shards
STICK_TO_SHARD = True if CACHED_DECODING else False
SKIP_CACHED_IMAGES = True if CACHED_DECODING else False

READ_SHUFFLE = args.read_shuffle

DISABLE_MMAP = args.disable_mmap

SMALL_DATA_SET = args.small
if SMALL_DATA_SET:
    test_data = small_test_data

print(
    f"GPUs: {N}, batch: {BATCH_SIZE}, workers: {WORKERS}, prefetch depth: {PREFETCH}, "
    f"loging interval: {LOG_INTERVAL}, fp16: {FP16}, NHWC: {NHWC}, READ_SHUFFLE: {READ_SHUFFLE}, "
    f"DISABLE_MMAP: {DISABLE_MMAP}, small dataset: {SMALL_DATA_SET}, GPU ID: {GPU_ID}, "
    f"shard number: {DALI_SHARD}, number of shards {NUMBER_OF_SHARDS}"
)

for pipe_name in test_data.keys():
    data_set_len = len(test_data[pipe_name])
    for i, data_set in enumerate(test_data[pipe_name]):
        if GPU_ID is None:
            pipes = [
                pipe_name(
                    batch_size=BATCH_SIZE,
                    num_threads=WORKERS,
                    device_id=n,
                    num_shards=NUMBER_OF_SHARDS,
                    data_paths=data_set,
                    prefetch=PREFETCH,
                    fp16=FP16,
                    random_shuffle=READ_SHUFFLE,
                    dont_use_mmap=DISABLE_MMAP,
                    nhwc=NHWC,
                    decoder_type=DECODER_TYPE,
                    decoder_cache_params=DECODER_CACHE_PARAMS,
                    reader_queue_depth=READER_QUEUE_DEPTH,
                    shard_id=DALI_SHARD + n,
                )
                for n in range(N)
            ]
        else:
            pipes = [
                pipe_name(
                    batch_size=BATCH_SIZE,
                    num_threads=WORKERS,
                    device_id=GPU_ID,
                    num_shards=NUMBER_OF_SHARDS,
                    data_paths=data_set,
                    prefetch=PREFETCH,
                    fp16=FP16,
                    random_shuffle=READ_SHUFFLE,
                    dont_use_mmap=DISABLE_MMAP,
                    nhwc=NHWC,
                    decoder_type=DECODER_TYPE,
                    decoder_cache_params=DECODER_CACHE_PARAMS,
                    reader_queue_depth=READER_QUEUE_DEPTH,
                    shard_id=DALI_SHARD,
                )
            ]
        [pipe.build() for pipe in pipes]

        if args.iters < 0:
            iters = pipes[0].epoch_size("Reader")
            assert all(pipe.epoch_size("Reader") == iters for pipe in pipes)
            iters_tmp = iters
            iters = iters // BATCH_SIZE
            if iters_tmp != iters * BATCH_SIZE:
                iters += 1
            iters_tmp = iters

            iters = iters // NUMBER_OF_SHARDS
            if iters_tmp != iters * NUMBER_OF_SHARDS:
                iters += 1
        else:
            iters = args.iters

        print("RUN {0}/{1}: {2}".format(i + 1, data_set_len, pipe_name.__name__))
        print(data_set)
        end = time.time()
        for i in range(args.epochs):
            if i == 0:
                print("Warm up")
            data_time = AverageMeter()
            for j in range(iters):
                for pipe in pipes:
                    pipe.run()
                data_time.update(time.time() - end)
                if j % LOG_INTERVAL == 0:
                    print(
                        f"System {SYS_ID}, GPU {GPU_ID}, run {i}: "
                        f" {pipe_name.__name__} {j + 1}/ {iters}, "
                        f"avg time: {data_time.avg} [s], "
                        f"worst time: {data_time.max_val} [s], "
                        f"speed: {N * BATCH_SIZE / data_time.avg} [img/s]"
                    )
                end = time.time()

        print("OK {0}/{1}: {2}".format(i, data_set_len, pipe_name.__name__))
