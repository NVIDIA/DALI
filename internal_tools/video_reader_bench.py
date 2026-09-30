# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""Benchmark for nvidia.dali.fn.readers.video and nvidia.dali.fn.experimental.readers.video.

Mirrors the CLI conventions and "Total Throughput: X frames/sec" reporting line of
internal_tools/hw_decoder_bench.py (used by qa/TL1_decoder_perf), so it can be wired into an
equivalent qa/TL1_video_reader_perf job using the same extract_perf/perf_check pattern from
qa/TL1_decoder_perf/test.sh.
"""

import argparse
import glob
import os
import statistics
import time

import nvidia.dali.fn as fn
from nvidia.dali.pipeline import pipeline_def

parser = argparse.ArgumentParser(description="DALI video reader benchmark")
parser.add_argument("-b", dest="batch_size", help="batch size", default=8, type=int)
parser.add_argument("-d", dest="device_id", help="device id", default=0, type=int)
parser.add_argument(
    "-g",
    dest="device",
    choices=["gpu", "cpu"],
    help="device to use (legacy reader is GPU-only)",
    default="gpu",
    type=str,
)
parser.add_argument("-w", dest="warmup_iterations", help="warmup iterations", default=5, type=int)
parser.add_argument(
    "-t", dest="total_samples", help="total samples (batches worth) to read", default=300, type=int
)
parser.add_argument("-j", dest="num_threads", help="CPU threads", default=4, type=int)
input_files_arg = parser.add_mutually_exclusive_group(required=True)
input_files_arg.add_argument(
    "-i", dest="video_dir", help="directory of video files used for the benchmark"
)
input_files_arg.add_argument(
    "--video_list",
    dest="video_list",
    nargs="+",
    default=[],
    help="explicit list of video files used for the benchmark",
)
parser.add_argument(
    "-p",
    dest="reader",
    choices=["legacy", "experimental"],
    help="which reader to benchmark",
    default="experimental",
    type=str,
)
parser.add_argument(
    "--sequence_length", dest="sequence_length", help="frames per sequence", default=16, type=int
)
parser.add_argument(
    "--dtype",
    dest="dtype",
    choices=["uint8", "float"],
    default="uint8",
    help="output dtype (experimental reader only; legacy always benchmarks the equivalent arg)",
)
parser.add_argument(
    "--print_every_n_iterations",
    dest="print_every_n_iterations",
    help="If > 0, print statistics every N iterations.",
    default=-1,
    type=int,
)

args = parser.parse_args()

if args.reader == "legacy" and args.device == "cpu":
    parser.error("legacy reader (readers.video) is GPU-only")

video_files = args.video_list or sorted(glob.glob(os.path.join(args.video_dir, "*")))
if not video_files:
    parser.error(
        f"No video files found (video_dir={args.video_dir!r}, video_list={args.video_list!r})"
    )


@pipeline_def(
    batch_size=args.batch_size, num_threads=args.num_threads, device_id=args.device_id, seed=0
)
def LegacyVideoReaderPipeline():
    import nvidia.dali.types as types

    dtype = types.FLOAT if args.dtype == "float" else types.UINT8
    return fn.readers.video(
        device="gpu",
        filenames=video_files,
        sequence_length=args.sequence_length,
        random_shuffle=True,
        initial_fill=min(16, len(video_files)),
        dtype=dtype,
    )


@pipeline_def(
    batch_size=args.batch_size, num_threads=args.num_threads, device_id=args.device_id, seed=0
)
def ExperimentalVideoReaderPipeline():
    import nvidia.dali.types as types

    dtype = types.FLOAT if args.dtype == "float" else types.UINT8
    return fn.experimental.readers.video(
        device=args.device,
        filenames=video_files,
        sequence_length=args.sequence_length,
        random_shuffle=True,
        initial_fill=min(16, len(video_files)),
        dtype=dtype,
    )


print(f"Reader: {args.reader}")
print(f"Device: {args.device}")
print(f"Batch size: {args.batch_size}")
print(f"Sequence length: {args.sequence_length}")
print(f"dtype: {args.dtype}")
print(f"CPU threads: {args.num_threads}")
print(f"Video files: {len(video_files)}")

pipe = LegacyVideoReaderPipeline() if args.reader == "legacy" else ExperimentalVideoReaderPipeline()
pipe.build()

for _ in range(args.warmup_iterations):
    pipe.schedule_run()
    _ = pipe.share_outputs()
    pipe.release_outputs()
print("Warmup finished")

test_iterations = args.total_samples // args.batch_size
print("Test iterations: ", test_iterations)

start_time = time.perf_counter()
execution_times = []
for iteration in range(test_iterations):
    iter_start_time = time.perf_counter()
    pipe.schedule_run()
    _ = pipe.share_outputs()
    pipe.release_outputs()
    iter_end_time = time.perf_counter()
    execution_times.append(iter_end_time - iter_start_time)

    if args.print_every_n_iterations > 0 and (
        (iteration + 1) % args.print_every_n_iterations == 0 or iteration == test_iterations - 1
    ):
        elapsed_time = time.perf_counter() - start_time
        samples_throughput = (iteration + 1) * args.batch_size / elapsed_time
        frames_throughput = samples_throughput * args.sequence_length
        mean_t = statistics.mean(execution_times)
        median_t = statistics.median(execution_times)
        min_t = min(execution_times)
        max_t = max(execution_times)
        print(
            f"Iteration {iteration + 1}/{test_iterations} - "
            + f"Throughput: {frames_throughput:.2f} frames/sec "
            + f"(mean={mean_t:.6f}sec, median={median_t:.6f}sec, "
            + f"min={min_t:.6f}sec, max={max_t:.6f}sec)"
        )

end_time = time.perf_counter()
total_time = end_time - start_time
total_samples_throughput = test_iterations * args.batch_size / total_time
total_frames_throughput = total_samples_throughput * args.sequence_length
avg_t = statistics.mean(execution_times)
stdev_t = statistics.stdev(execution_times) if len(execution_times) > 1 else 0.0
median_t = statistics.median(execution_times)
min_t = min(execution_times)
max_t = max(execution_times)

print("\nFinal Results:")
print(f"Total Execution Time: {total_time:.6f} sec")
print(f"Total Samples Throughput: {total_samples_throughput:.2f} samples/sec")
print(f"Total Throughput: {total_frames_throughput:.2f} frames/sec")
print(f"Average time per iteration: {avg_t:.6f} sec")
print(f"Median time per iteration: {median_t:.6f} sec")
print(f"Stddev time per iteration: {stdev_t:.6f} sec")
print(f"Min time per iteration: {min_t:.6f} sec")
print(f"Max time per iteration: {max_t:.6f} sec")
