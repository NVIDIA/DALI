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

"""Benchmark of fn.experimental.readers.video, using DALI_extra data only.

  video_reader_bench.py --json a.json            throughput (frames/s) of every configuration
  video_reader_bench.py --json a.json --check    ... and fail if more threads (or a deeper reader
                                                 queue) are more than 10% slower than fewer
  video_reader_bench.py --json a.json --hash     checksums of the output instead of throughput
  video_reader_bench.py --compare a.json b.json  two builds: checksums equal, throughput ratio

Configurations (a group is one set of samples, read with different num_threads / reader_depth):

  lanes          256 distinct files, one sequence each: every sample is a new file, so the cost
                 of opening a file and the parallelism of decoding dominate
  cpu            the same on the CPU backend (VP8/VP9 files)
  seq, random    the 15 H.264 clips as they are, read in order or shuffled
  random_hevc    the HEVC trailer, shuffled (keyframe seeks, open GOPs)
  reader_depth   the reader's own prefetch_queue_depth at batch 1
  single_*       one long file, batch 1: a single stream, so what is measured is how fast one
                 decoder runs per frame, not how many run in parallel

The GPU configurations also run through torchcodec on the GPU (groups *_torchcodec, FFmpeg's
NVDEC), on the same samples: for reference only, never checked, skipped if it is not installed.

Every configuration runs in a fresh process, so the frame index and NVDEC decoder caches do not
carry over, and is repeated --repeat times: the median is reported. The first batches (those
the pipeline and the reader's queue may have produced before the clock starts) are not timed.

With --hash, the output of every configuration of a group must be identical, whatever
num_threads and reader_depth: each configuration records a checksum of the first >= 40 samples
(so that the reader's prefetch queue wraps around). The configurations marked "checksums only"
add reader_depth 2 with batches of 16, which the timed ones do not run.
"""

import argparse
import contextlib
import glob
import hashlib
import itertools
import json
import os
import random
import statistics
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor

VIDEO_DIR = "db/video"
# file sets: glob patterns relative to DALI_extra/db/video
CLIPS = ["sintel/video_files/*.mp4", "sintel/labelled_videos/*/*.mp4"]  # 15 H.264 720p clips
H264_LONG = ["sintel/sintel_trailer-720p.mp4"]  # 1253 frames, GOP 250
HEVC_LONG = ["hevc/sintel_trailer-720p.mp4"]  # the same clip as HEVC, a keyframe every ~9 frames
VPX = ["vp9/vp9_0.mp4", "vp9/vp9_2.mp4", "vp8/vp8.webm", "cfr/test_1_vp9.mp4", "vfr/test_1_vp9.mp4"]
HASH_MIN_SAMPLES = 40
PIPELINE_DEPTH = 2  # the pipeline's own prefetch queue, set explicitly


def dali_configs(threads):
    """(group, configuration) of the DALI reader.

    A configuration is a dict of: files (glob patterns), device, t (num_threads), rd (the reader's
    prefetch_queue_depth), batch (batch_size), seq (sequence_length), shuffle (random_shuffle),
    many (optional: that number of symlinks to the files, one sequence each; without it, the
    files as they are), hash_only (optional: not timed)."""
    base = dict(batch=16, seq=32, rd=1, shuffle=False, device="gpu")
    t_max = max(threads)
    for t in threads:
        yield "lanes", dict(base, files=CLIPS, many=256, t=t)
        yield "cpu", dict(base, files=VPX, many=128, t=t, device="cpu", batch=8, seq=16)
    for name, files in (("seq", CLIPS), ("random", CLIPS), ("random_hevc", HEVC_LONG)):
        for t in sorted({1, t_max}):
            yield name, dict(base, files=files, t=t, shuffle=name != "seq")
    for rd in (1, 2, 4):
        yield "reader_depth", dict(base, files=CLIPS, many=256, t=8, rd=rd, batch=1)
    for name, files in (("single_h264", H264_LONG), ("single_hevc", HEVC_LONG)):
        yield name, dict(base, files=files, t=4, batch=1, seq=64)
    # checksums only: several decode lanes with two batches in the reader's queue
    for name, extra in (("lanes", dict(many=256)), ("seq", {}), ("random", dict(shuffle=True))):
        yield name, dict(base, files=CLIPS, t=t_max, rd=2, hash_only=True, **extra)


def configs(threads, torchcodec=True):
    cfgs = list(dali_configs(threads))
    yield from cfgs
    for group, cfg in cfgs if torchcodec else ():
        if cfg["device"] == "gpu" and group != "reader_depth" and not cfg.get("hash_only"):
            yield group + "_torchcodec", dict(cfg, lib="torchcodec")


def key(group, cfg):
    return f"{group} t={cfg['t']} rd={cfg['rd']}"


def find_files(cfg, root):
    files = sorted(f for p in cfg["files"] for f in glob.glob(os.path.join(root, VIDEO_DIR, p)))
    if not files:
        raise FileNotFoundError(
            f"--dali_extra={root!r}: expected files matching {cfg['files']} under "
            f"{os.path.join(root, VIDEO_DIR)!r}, but found none. Point --dali_extra (or "
            "DALI_EXTRA_PATH) at a DALI_extra checkout."
        )
    return files


def torchcodec_step(cfg, samples):
    """Returns a function that decodes a batch of (path, start, end) samples with torchcodec on
    min(t, batch) threads, each keeping its decoder while consecutive samples are of one file."""
    import torch
    from torchcodec import decoders

    if cfg["shuffle"]:
        random.Random(0).shuffle(samples)
    batches, local, keep = itertools.cycle(samples), threading.local(), []
    pool = ThreadPoolExecutor(min(cfg["t"], cfg["batch"]))
    # FFmpeg's NVDEC, as the default CUDA backend differs between torchcodec versions
    backend = getattr(decoders, "set_cuda_backend", None)

    def decode(sample):
        path, first, end = sample
        if getattr(local, "path", None) != path:
            torch.cuda.set_device(0)
            with backend("ffmpeg") if backend else contextlib.nullcontext():
                local.decoder = decoders.VideoDecoder(path, device="cuda:0")
            local.path = path
        frames = local.decoder.get_frames_in_range(first, end).data
        assert len(frames) == end - first, f"{path}[{first}:{end}]: {len(frames)} frames"
        return frames

    def run_batch():
        keep[:] = pool.map(decode, [next(batches) for _ in range(cfg["batch"])])  # as DALI's
        torch.cuda.synchronize()

    torch.zeros(1, device="cuda")  # the CUDA context, before the clock starts
    return run_batch


def child(cfg, root, hash_out):
    """Runs one configuration in this process: {"fps", "build_s"}, or {"hash"}."""
    with tempfile.TemporaryDirectory() as tmp:
        return run_config(cfg, root, hash_out, tmp)


def run_config(cfg, root, hash_out, tmp):
    import numpy as np

    files = find_files(cfg, root)
    samples = []  # (path, start, end)
    if cfg.get("many"):  # distinct symlinks: nothing keyed by the file name is shared
        lines = []
        for i in range(cfg["many"]):
            link = os.path.join(tmp, f"{i}_{os.path.basename(files[i % len(files)])}")
            os.symlink(files[i % len(files)], link)
            lines.append(f"{link} {i} 0 {cfg['seq']}\n")  # frames [0, seq)
            samples.append((link, 0, cfg["seq"]))
        with open(os.path.join(tmp, "list.txt"), "w") as f:
            f.writelines(lines)
        src = dict(
            file_list=os.path.join(tmp, "list.txt"),
            file_list_format="frames",
            file_list_include_end=False,
        )
    else:
        src = dict(filenames=files, labels=list(range(len(files))))

    t0 = time.perf_counter()
    # run_batch() produces one batch: the DALI pipeline's run, or the same work in torchcodec
    if cfg.get("lib") == "torchcodec":
        if not cfg.get("many"):  # whole files in full sequences, as the DALI reader splits them
            from torchcodec.decoders import VideoDecoder

            for f in files:
                for first in range(0, len(VideoDecoder(f)) - cfg["seq"] + 1, cfg["seq"]):
                    samples.append((f, first, first + cfg["seq"]))
        run_batch = torchcodec_step(cfg, samples)
        epoch_size = len(samples)
    else:
        import nvidia.dali.fn as fn
        from nvidia.dali.pipeline import pipeline_def

        @pipeline_def(
            batch_size=cfg["batch"],
            num_threads=cfg["t"],
            device_id=0,
            seed=42,
            prefetch_queue_depth=PIPELINE_DEPTH,
        )
        def pipe():
            return tuple(
                fn.experimental.readers.video(
                    device=cfg["device"],
                    sequence_length=cfg["seq"],
                    random_shuffle=cfg["shuffle"],
                    prefetch_queue_depth=cfg["rd"],
                    **src,
                )
            )

        p = pipe()
        p.build()
        run_batch = p.run
        epoch_size = sum(p.epoch_size().values())
    build_s = time.perf_counter() - t0

    if hash_out:  # from batch 0: the same samples whatever the queue depths
        h = hashlib.sha1()
        # more than an epoch, so that the reader's restart is covered too
        n_samples = max(HASH_MIN_SAMPLES, epoch_size)
        for _ in range(-(-n_samples // cfg["batch"]) + 2):
            for output in run_batch():
                output = output.as_cpu() if hasattr(output, "as_cpu") else output
                for i in range(len(output)):
                    h.update(np.ascontiguousarray(np.array(output[i])).tobytes())
        return dict(hash=h.hexdigest())
    # batches that may have been produced before the clock starts: pipeline + reader queues
    for _ in range(PIPELINE_DEPTH + cfg["rd"] + 1):
        run_batch()
    n, t0 = 0, time.perf_counter()
    while n < 20 or time.perf_counter() - t0 < 5:  # >= 20 batches and >= 5 s
        run_batch()
        n += 1
    return dict(fps=n * cfg["batch"] * cfg["seq"] / (time.perf_counter() - t0), build_s=build_s)


def check_scaling(out, tolerance=0.10):
    """More threads, or a deeper reader queue, must not be slower than fewer (flat is fine)."""
    ok = True
    for group in sorted({v["group"] for v in out.values() if "fps" in v}):
        rows = sorted((v["t"], v["rd"], v["fps"]) for v in out.values() if v["group"] == group)
        if group.endswith("_torchcodec") or len(rows) < 2:
            continue
        best = rows[0]  # the fastest of the rows with fewer threads / a shallower queue
        slower = []
        for row in rows[1:]:
            if row[2] < (1 - tolerance) * best[2]:
                slower.append((best, row))
            best = max(best, row, key=lambda r: r[2])
        ok &= not slower
        for (t0, rd0, slow), (t1, rd1, fast) in slower or [(rows[0], rows[-1])]:
            print(
                f"{'FAIL' if slower else 'OK':4} {group}: {slow:.0f} frames/s (t={t0}, rd={rd0})"
                f" -> {fast:.0f} (t={t1}, rd={rd1})"
            )
    return ok


def run(args):
    todo = [
        (group, cfg)
        for group, cfg in configs(args.threads, args.torchcodec and not args.hash)
        if args.only in group and (args.hash or not cfg.get("hash_only"))
    ]
    for _, cfg in todo:  # once here, not as a failure of every child process
        find_files(cfg, args.dali_extra)
    out = {}
    for group, cfg in todo:
        reference = cfg.get("lib") == "torchcodec"  # never fails the run
        reps = []
        for _ in range(1 if args.hash else args.repeat):
            r = subprocess.run(
                [
                    sys.executable,
                    __file__,
                    "--child",
                    json.dumps(cfg),
                    "--dali_extra",
                    args.dali_extra,
                ]
                + ["--hash"] * args.hash,
                capture_output=True,
                text=True,
            )
            if r.returncode:
                reps = None
                print(f"FAILED {key(group, cfg)}{' (reference)' if reference else ''}:")
                print(r.stderr[-1500:])
                break
            reps.append(json.loads(r.stdout.splitlines()[-1]))
        if reps:
            out[key(group, cfg)] = dict(group=group, t=cfg["t"], rd=cfg["rd"])
            for k in reps[0]:
                out[key(group, cfg)][k] = (
                    reps[0][k] if k == "hash" else statistics.median(x[k] for x in reps)
                )
            print(key(group, cfg), {k: v for k, v in out[key(group, cfg)].items() if k != "group"})
    failed = any(
        key(g, cfg) not in out for g, cfg in todo if cfg.get("lib") != "torchcodec"
    )  # a failed or missing configuration
    for g in {v["group"] for v in out.values()} if args.hash else ():
        # one group, one output, whatever num_threads / reader_depth
        if len({v["hash"] for v in out.values() if v["group"] == g}) > 1:
            print(f"MISMATCH within group {g}")
            failed = True
    if args.check and not args.hash:
        failed |= not check_scaling(out)
    with open(args.json, "w") as f:
        json.dump(out, f, indent=1)
    return int(failed)


def compare(path_a, path_b):
    with open(path_a) as f:
        a = json.load(f)
    with open(path_b) as f:
        b = json.load(f)
    # a configuration missing on one side is a failure (the torchcodec rows are for reference)
    bad = [k for k in a.keys() ^ b.keys() if "_torchcodec" not in k]
    for k in sorted(a.keys() & b.keys()):
        if a[k].get("hash") != b[k].get("hash"):
            bad.append(k)
        if "fps" in a[k] and "fps" in b[k]:
            ratio = b[k]["fps"] / a[k]["fps"]
            print(f"{k:34} {a[k]['fps']:9.1f} -> {b[k]['fps']:9.1f} frames/s  x{ratio:.2f}")
    print("DIFFERENT / MISSING:", bad if bad else "none")
    return int(bool(bad))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--json", default="video_reader_bench.json", help="where to write the results")
    ap.add_argument("--hash", action="store_true", help="checksums instead of throughput")
    ap.add_argument("--check", action="store_true", help="fail if threads / depth slow it down")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"), help="two result files")
    ap.add_argument(
        "--threads", type=lambda s: [int(x) for x in s.split(",")], default=[1, 2, 4, 8]
    )
    ap.add_argument("--repeat", type=int, default=3, help="fresh-process runs per configuration")
    ap.add_argument("--only", default="", help="only groups containing this text")
    ap.add_argument("--no_torchcodec", dest="torchcodec", action="store_false", help="DALI only")
    ap.add_argument("--dali_extra", default=os.environ.get("DALI_EXTRA_PATH", ""))
    ap.add_argument("--child", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.dali_extra:  # symlink targets and file lists must not depend on the working directory
        args.dali_extra = os.path.abspath(args.dali_extra)
    if args.child:
        print(json.dumps(child(json.loads(args.child), args.dali_extra, args.hash)))
    elif args.compare:
        sys.exit(compare(*args.compare))
    else:
        sys.exit(run(args))


if __name__ == "__main__":
    main()
