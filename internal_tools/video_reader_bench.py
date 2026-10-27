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

"""Benchmark for fn.experimental.readers.video, using DALI_extra data only.

Scenarios (each isolates one aspect of the reader):

  lanes          many distinct files, one keyframe-aligned sequence each, so every
                 batch holds `batch` different files and no sample needs a seek:
                 per-file cost (opening, indexing, decoder setup, decoding the lead-in
                 frames an edit list hides, see LEAD_IN_FILES) plus decoding, vs.
                 num_threads. A per-file speedup also shows here; check `cache_key`
                 and `engines` before attributing a gain to decode parallelism
  seq_longgop    15 H.264 clips read sequentially (long GOPs, consecutive sequences
                 of the same file): no-regression check for sequential access
  random_access  the same clips, and the HEVC trailer (open GOPs), shuffled:
                 parallelism and keyframe-seek waste together (do not read a lanes
                 speedup from this one alone)
  single_stream  one clip, batch 1: per-stream throughput, i.e. per-frame
                 serialization. Includes a 512x512 clip, where per-frame overheads
                 matter more than decode time
  reader_depth   batch 1, the reader's own prefetch_queue_depth (batches in flight),
                 on many distinct files and on one sequential long-GOP file
  cache_key      every sample switches between two files with the same format, either
                 cropped (800x600 coded as 800x608) or not (1280x720), vs. a control
                 that rewinds the same file to frame 0 for every sample instead:
                 the cost of opening another file (decoder creation or NVDEC decoder
                 cache reuse) relative to a flush and seek within the same file. This
                 is a workload difference, not an isolated switch cost
  engines        K independent readers in one pipeline, each on its own files:
                 how throughput scales with concurrent NVDEC sessions (works today,
                 because every reader decodes on its own thread)
  cpu            many distinct VP8/VP9 files on the CPU backend vs. num_threads

Methodology:
  * every configuration runs in a fresh subprocess (no cache carry-over), --repeat
    times, round-robin over configurations; the median and the spread are reported
  * the first `depth` + `reader_depth` timed iterations are excluded, as up to that many
    batches (pipeline queue and the reader's own queue) may have been produced off the
    clock; measuring continues until both --iters and --min_time are reached after them
  * throughput counts the actual output frames of the timed batches
  * Pipeline.run() without cuda_stream returns host-synchronized outputs, so iteration
    times include GPU completion
  * NVML metrics are per process where possible (decoder utilization, memory). A
    result is flagged as contaminated if other processes use the GPU or NVDEC is busy
    before the run starts. Decoder utilization (dec%) is averaged over the GPU's NVDEC
    engines, so on GPUs with several engines it is low even for a fully busy session
  * all input files are read once before the first run, so all runs see a warm page cache
  * DALI_extra has no 1080p content, so absolute rates are lower than for 1080p data;
    compare builds, not datasets

--hash records a checksum of every sample (frames, label, frame_num) of at least
HASH_MIN_SAMPLES samples, so that the reader's prefetch queue wraps around, and with
--hash_epochs E of E epochs and one more batch, so that decoders are reused across
batches and the reader wraps around its epochs. Compare two --hash result files (e.g. two
builds, or --ignore threads to check that the output does not depend on num_threads) with
--compare. --hash also runs the configurations only checksummed:
  lanes_depth    batch 16 on several decode lanes with the reader's prefetch_queue_depth
                 1 and 2: whole files in order and shuffled, and many distinct files

Other readers and libraries (--backends), for reference:
  dali_legacy      DALI's legacy fn.readers.video (GPU, constant frame rate only: no cpu,
                   cache_key), through the same pipeline as the DALI reader
  dali_decoder     fn.readers.file + fn.decoders.video (fn.experimental.decoders.video in
                   older DALI): every sample is a whole file, read by the file reader, and
                   decoded with a new decoder per sample, the samples of a batch in parallel
                   on num_threads threads. Same samples (file, first and last frame) as the
                   DALI reader, shuffled by the file reader; `reader_depth` is the file
                   reader's prefetch_queue_depth
  pynvvideocodec   PyNvVideoCodec SimpleDecoder (NVDEC), RGB interleaved output (as DALI)
  torchcodec       torchcodec VideoDecoder, FFmpeg CUDA backend on GPU, FFmpeg on CPU
  torchcodec_nvdec torchcodec VideoDecoder with its own NVDEC backend (torchcodec >= 0.8)
  They decode the same samples as the DALI reader (the same sequences of the same files,
  in the same order, or shuffled with a fixed seed), each sample with frames [start, end)
  of one file. A batch is decoded by a pool of min(threads, batch) threads, each with its
  own decoder, kept while consecutive samples come from the same file (PyNvVideoCodec
  reconfigures its decoder for a new file, torchcodec opens a new one, scanning it, as
  its default seek_mode="exact" does). Batches are not prefetched: unlike DALI, a batch
  is decoded when it is requested. A batch's outputs are kept until the next batch is
  requested, as DALI's, so the process memory (mem MB) includes one batch for them, and
  for DALI also the batches in its queues (`depth` + `reader_depth`). PyNvVideoCodec is an
  exception: its frames stay in its decoder's buffers, which the decoder's next call
  reuses, so its memory does not include a batch. They run only the configurations that
  mean the same for them (backend_runs), and need torch. Rows are flagged when torchcodec
  reports a CPU fallback, and PyNvVideoCodec and dali_decoder rows on clips with hidden
  lead-in frames (LEAD_IN_FILES), where they return different frames.
"""

import argparse
import glob
import hashlib
import json
import os
import random
import shutil
import socket
import statistics
import subprocess
import sys
import tempfile
import threading
import time
import warnings

VIDEO_DIR = "db/video"

# --hash checks at least this many samples per configuration: twice the batches the
# pipeline and the reader can hold (depth 2 + reader_depth up to 8) at batch 1, so that
# every slot of the reader's prefetch queue is reused
HASH_MIN_SAMPLES = 40

VPX_CLIPS = [
    "vp9/vp9_0.mp4",
    "vp9/vp9_2.mp4",
    "vp8/vp8.webm",
    "cfr/test_1_vp9.mp4",
    "cfr/test_2_vp9.mp4",
    "vfr/test_1_vp9.mp4",
    "vfr/test_2_vp9.mp4",
]
# file sets: glob patterns relative to DALI_extra/db/video
FILES = {
    # 15 H.264 1280x720 clips, 122-430 frames, B-frames, 1-4 keyframes (GOP up to 250)
    "h264_clips": ["sintel/video_files/*.mp4", "sintel/labelled_videos/*/*.mp4"],
    # one H.264 1280x720 clip, 1253 frames, GOP 250
    "h264_long": ["sintel/sintel_trailer-720p.mp4"],
    # the same clip as HEVC 8-bit (keyframe every ~9 frames) and HEVC 10-bit (GOP 250)
    "hevc_long": ["hevc/sintel_trailer-720p.mp4"],
    "hevc10_long": ["hevc/sintel_trailer-720p_10b.mp4"],
    # one H.264 512x512 clip, 250 frames: small frames, per-frame overheads dominate
    "h264_small": ["test_label.mp4"],
    # two H.264 800x600 clips, coded as 800x608 (display area cropped)
    "h264_cropped_pair": ["cfr/test_2.mp4", "vfr/test_2.mp4"],
    # two H.264 1280x720 clips, coded size == display size
    "h264_aligned_pair": ["cfr_test.mp4", "vfr_test.mp4"],
    # VP8/VP9 clips (H.264 is not enabled on the CPU backend)
    "vpx_clips": VPX_CLIPS,
}
# File sets where most clips (cut without re-encoding) start before the frame they show first,
# with an MP4 edit list hiding the lead-in frames: the DALI reader and torchcodec start at the
# first visible frame, so they decode the lead-in before frame 0, while PyNvVideoCodec and
# DALI's video decoder (dali_decoder) return the lead-in frames themselves (different frames,
# less decoding)
LEAD_IN_FILES = {"h264_clips"}
LEAD_IN_BACKENDS = {"pynvvideocodec", "dali_decoder"}


def _cfg(files, layout="plain", device="gpu", batch=8, seq=32, threads=4, **kw):
    """A benchmark configuration. Only JSON-serializable values.

    layout: "plain"     - the files as they are (filenames=...)
            "many:N"    - N distinct symlinks cycling over the files, each with one
                          sequence starting at frame 0 (a keyframe)
            "alternate:N" - N lines alternating between the files, each starting at
                          frame 0, read in order: every sample switches files
            "same:N"    - N lines, all of the first file, each starting at frame 0: the
                          control for "alternate" (no file switches, but every sample
                          after the first rewinds the same decoder to frame 0)
    """
    cfg = dict(
        backend="dali",
        files=files,
        layout=layout,
        device=device,
        batch=batch,
        seq=seq,
        threads=threads,
        depth=2,  # pipeline prefetch_queue_depth (DALI default)
        reader_depth=1,  # the reader's own prefetch_queue_depth (DALI default)
        readers=1,
        shuffle=False,
    )
    cfg.update(kw)
    return cfg


def scenarios(threads, readers, quick=False):
    """The configurations of every scenario (HASH_ONLY_SCENARIOS only with --hash)."""
    t_max = max(threads)
    t_pair = sorted({1, t_max})
    iters = {} if quick else dict(iters=20)
    return {
        "lanes": [_cfg("h264_clips", "many:256", batch=16, seq=32, threads=t) for t in threads],
        "seq_longgop": [_cfg("h264_clips", batch=16, seq=32, threads=t) for t in t_pair],
        "random_access": [
            _cfg(f, batch=16, seq=32, threads=t, shuffle=True)
            for f in ("h264_clips", "hevc_long")
            for t in t_pair
        ],
        "single_stream": [
            _cfg(f, batch=1, seq=64, threads=4)
            for f in ("h264_long", "hevc_long", "hevc10_long", "h264_small")
        ],
        "reader_depth": [
            _cfg(f, layout, batch=1, seq=32, threads=8, reader_depth=d)
            for f, layout in (("h264_clips", "many:256"), ("h264_long", "plain"))
            for d in (1, 2, 4, 8)
        ],
        # ms_per_sample(alternate) - ms_per_sample(same), per file pair: cost of opening the
        # other file relative to rewinding the same one (see the module docstring). One
        # thread: with more, other libraries keep each file of a pair on its own decoder
        "cache_key": [
            _cfg(f, f"{kind}:64", batch=8, seq=8, threads=1)
            for f in ("h264_cropped_pair", "h264_aligned_pair")
            for kind in ("alternate", "same")
        ],
        # the noisiest scenario: more iterations than the --iters default
        "engines": [
            _cfg("h264_clips", "many:64", batch=4, seq=32, threads=4, readers=k, **iters)
            for k in readers
        ],
        "cpu": [
            _cfg("vpx_clips", "many:128", device="cpu", batch=8, seq=16, threads=t) for t in threads
        ],
        # several decode lanes with up to two batches in the reader's queue (the other
        # scenarios with several lanes use reader_depth 1)
        "lanes_depth": [
            _cfg(f, layout, batch=16, seq=32, threads=t, reader_depth=d, shuffle=s)
            for f, layout, s in (
                ("h264_clips", "plain", False),
                ("h264_clips", "plain", True),
                ("h264_clips", "many:256", False),
            )
            for d in (1, 2)
            for t in threads
        ],
    }


# scenarios only checksummed (--hash), not measured
HASH_ONLY_SCENARIOS = {"lanes_depth"}


# scenarios that other backends run as well (reader_depth and engines measure properties of
# the DALI reader), and the devices they decode on
_REFERENCE_SCENARIOS = {"lanes", "seq_longgop", "random_access", "single_stream", "cache_key"}
BACKEND_SCENARIOS = {
    "dali": (None, ("gpu", "cpu")),  # None: all scenarios
    # the legacy fn.readers.video: GPU only, and constant frame rate only (both cache_key
    # pairs include a variable frame rate file)
    "dali_legacy": (_REFERENCE_SCENARIOS - {"cache_key"} | {"reader_depth", "engines"}, ("gpu",)),
    # fn.readers.file + fn.decoders.video; reader_depth would prefetch file contents only
    "dali_decoder": (_REFERENCE_SCENARIOS | {"engines", "cpu"}, ("gpu", "cpu")),
    "pynvvideocodec": (_REFERENCE_SCENARIOS, ("gpu",)),
    "torchcodec": (_REFERENCE_SCENARIOS | {"cpu"}, ("gpu", "cpu")),
    "torchcodec_nvdec": (_REFERENCE_SCENARIOS, ("gpu",)),
}


def backend_runs(backend, scenario, cfg):
    names, devices = BACKEND_SCENARIOS[backend]
    if (names is not None and scenario not in names) or cfg["device"] not in devices:
        return False
    # PyNvVideoCodec 2.0 SimpleDecoder fails (IndexError) when a decoder that was just
    # created or reconfigured for a file is asked for frames from the middle of it, so it
    # runs only configurations whose decoders start every file at frame 0
    seeks = scenario == "random_access" or (
        scenario == "seq_longgop" and min(cfg["threads"], cfg["batch"]) > 1
    )
    if backend == "pynvvideocodec" and seeks:
        return False
    # the legacy reader fails seeking in the HEVC trailer (open GOPs): "The decoder returned a
    # frame that is past the expected one"; reading it sequentially works
    return not (backend == "dali_legacy" and seeks and cfg["files"] == "hevc_long")


# ----------------------------------------------------------------------------------------
# data


def resolve_files(dali_extra, name):
    # absolute paths: they are used as symlink targets in a temporary directory
    root = os.path.abspath(os.path.join(dali_extra, VIDEO_DIR))
    files, missing = [], []
    for pattern in FILES[name]:
        matches = sorted(glob.glob(os.path.join(root, pattern)))
        if not matches:
            missing.append(pattern)
        files += matches
    if missing:
        raise FileNotFoundError(
            f"File set '{name}': no files match {missing} under '{root}'. "
            f"Check that --dali_extra (or DALI_EXTRA_PATH) points to a DALI_extra checkout."
        )
    return files


def materialize(cfg, dali_extra, workdir):
    """Creates the inputs of a configuration. Returns one input spec per reader.

    A spec is either {"filenames": [...]} or {"file_list": path}. The result is fully
    determined by the configuration, so repeated runs (and --hash) see the same samples.
    """
    files = resolve_files(dali_extra, cfg["files"])
    kind, _, n = cfg["layout"].partition(":")
    if kind == "plain":
        if cfg["readers"] != 1:
            raise ValueError(f"layout='plain' requires readers=1; got readers={cfg['readers']}.")
        return [{"filenames": files}]
    if kind not in ("many", "alternate", "same") or not n.isdigit() or int(n) < 1:
        raise ValueError(
            f"Unknown layout {cfg['layout']!r}; expected 'plain', 'many:N', 'alternate:N' "
            f"or 'same:N' with a positive integer N."
        )
    n = int(n)
    seq = cfg["seq"]
    tag = "_".join(str(cfg[k]) for k in ("files", "layout", "seq", "readers"))
    out_dir = os.path.join(workdir, tag.replace(":", "-"))
    os.makedirs(out_dir, exist_ok=True)
    lines = []
    if kind == "many":
        for i in range(n * cfg["readers"]):
            src = files[i % len(files)]
            dst = os.path.join(out_dir, f"{i:05d}_{os.path.basename(src)}")
            if not os.path.lexists(dst):
                os.symlink(src, dst)
            lines.append((dst, i))
    elif kind == "alternate":
        lines = [(files[i % len(files)], i) for i in range(n)]
    else:  # same
        lines = [(files[0], i) for i in range(n)]
    specs = []
    per_reader = len(lines) // cfg["readers"]
    for r in range(cfg["readers"]):
        path = os.path.join(out_dir, f"file_list_{r}.txt")
        with open(path, "w") as f:
            for name, label in lines[r * per_reader : (r + 1) * per_reader]:
                # frames [0, seq) (file_list_include_end=False): exactly one sequence, starting
                # at frame 0, a keyframe, so no sample decodes frames it doesn't return
                f.write(f"{name} {label} 0 {seq}\n")
        specs.append({"file_list": path})
    return specs


def prewarm(paths):
    """Reads files once so that every run sees a warm page cache."""
    for p in sorted(set(paths)):
        with open(p, "rb") as f:
            while f.read(1 << 24):
                pass


# ----------------------------------------------------------------------------------------
# GPU monitoring


def nvml_handle(device_id):
    """NVML handle of the CUDA device `device_id`, honoring CUDA_VISIBLE_DEVICES.

    Assumes CUDA_DEVICE_ORDER=PCI_BUS_ID or a machine where CUDA and NVML enumerate
    devices the same way. Returns (pynvml, handle) or raises.
    """
    import pynvml

    pynvml.nvmlInit()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible:
        entry = visible.split(",")[device_id].strip()
        if entry.isdigit():
            return pynvml, pynvml.nvmlDeviceGetHandleByIndex(int(entry))
        # older nvidia-ml-py versions take and return bytes, newer ones accept both
        return pynvml, pynvml.nvmlDeviceGetHandleByUUID(entry.encode())
    return pynvml, pynvml.nvmlDeviceGetHandleByIndex(device_id)


def other_gpu_pids(nvml, handle):
    procs = nvml.nvmlDeviceGetComputeRunningProcesses(handle)
    procs += nvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
    return sorted({p.pid for p in procs if p.pid != os.getpid()})


class GpuMonitor:
    """Samples per-process GPU memory and collects per-process NVDEC utilization.

    While sampling, it also records any other process that uses the GPU, so that a
    competing process that starts and exits during a run is detected.
    """

    def __init__(self, device_id, period=0.1):
        self.errors = []  # NVML problems; a metric may be missing, the run is still valid
        self.period = period
        self.mem = []
        self.others = set()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        try:
            self.nvml, self.handle = nvml_handle(device_id)
        except Exception as e:
            self.nvml = None
            self._error(f"NVML unavailable: {e}")

    def _error(self, msg):
        if msg not in self.errors:
            self.errors.append(msg)

    def snapshot(self):
        """Pre/post-run state: other processes, NVDEC utilization."""
        if not self.nvml:
            return {}
        nvml, h = self.nvml, self.handle
        res = {}
        queries = dict(
            other_pids=lambda: other_gpu_pids(nvml, h),
            dev_nvdec_util=lambda: nvml.nvmlDeviceGetDecoderUtilization(h)[0],
        )
        for key, query in queries.items():
            try:  # older NVML bindings may lack some queries for newer drivers
                res[key] = query()
            except Exception as e:
                self._error(f"NVML {key} failed: {e}")
        return res

    def _run(self):
        nvml, h, pid = self.nvml, self.handle, os.getpid()

        def own_memory():
            for p in nvml.nvmlDeviceGetComputeRunningProcesses(h):
                if p.pid == pid and p.usedGpuMemory is not None:
                    self.mem.append(p.usedGpuMemory)

        samplers = {
            "own memory": own_memory,
            "other processes": lambda: self.others.update(other_gpu_pids(nvml, h)),
        }
        failures = dict.fromkeys(samplers, 0)
        while not self._stop.is_set():
            for name, sample in list(samplers.items()):
                try:
                    sample()
                    failures[name] = 0
                except Exception as e:
                    # e.g. a process list that changed size during the query; only give up
                    # on a query that keeps failing (unsupported by these bindings/driver)
                    failures[name] += 1
                    if failures[name] >= 3:
                        self._error(f"NVML sampling of {name} failed: {e}")
                        del samplers[name]
            time.sleep(self.period)

    def start(self):
        self._t_start_us = int(time.time() * 1e6)
        if self.nvml:
            self._thread.start()

    def stop(self):
        if not self.nvml:
            return {}
        self._stop.set()
        self._thread.join()
        res = {}
        if self.mem:
            res["proc_mem_max_mb"] = max(self.mem) / 2**20
        try:
            samples = self.nvml.nvmlDeviceGetProcessUtilization(self.handle, self._t_start_us)
            dec = [s.decUtil for s in samples if s.pid == os.getpid()]
            if dec:
                res["proc_nvdec_util"] = statistics.mean(dec)
        except Exception:
            # per-process utilization is optional: not every driver/binding supports it, or
            # there may be no samples yet; the result is still valid without it
            res["proc_nvdec_util"] = None
        return res


# ----------------------------------------------------------------------------------------
# running


def num_frames_dali(path, device, device_id):
    """The frames of a file as the DALI reader counts them: its sequences of one frame."""
    import nvidia.dali.fn as fn
    from nvidia.dali.pipeline import pipeline_def

    @pipeline_def(batch_size=1, num_threads=1, device_id=device_id)
    def count():
        return fn.experimental.readers.video(
            filenames=[path], sequence_length=1, device=device, name="Count"
        )

    p = count()
    p.build()
    return p.reader_meta("Count")["epoch_size"]


def build_pipeline(cfg, specs, device_id, seed, frame_num=False):
    import nvidia.dali.fn as fn
    from nvidia.dali.pipeline import pipeline_def

    legacy = cfg["backend"] == "dali_legacy"
    reader = fn.readers.video if legacy else fn.experimental.readers.video
    if cfg["backend"] == "dali_decoder":

        def num_frames(path):
            return num_frames_dali(path, cfg["device"], device_id)

        # the samples the DALI reader would produce, (path, start, end), one list per reader
        samples = [sample_list([s], cfg["seq"], False, num_frames) for s in specs]
        # moved out of experimental in DALI 2.0
        decoder = getattr(fn.decoders, "video", None) or fn.experimental.decoders.video

    @pipeline_def(
        batch_size=cfg["batch"],
        num_threads=cfg["threads"],
        device_id=device_id,
        prefetch_queue_depth=cfg["depth"],
        seed=seed,
    )
    def pipe():
        outs = []
        if cfg["backend"] == "dali_decoder":
            for r, reader_samples in enumerate(samples):
                # whole files, with the sample's first frame as the label
                encoded, start = fn.readers.file(
                    files=[path for path, _, _ in reader_samples],
                    labels=[start for _, start, _ in reader_samples],
                    random_shuffle=cfg["shuffle"],
                    prefetch_queue_depth=cfg["reader_depth"],
                    name=f"Reader{r}",
                )
                video = decoder(
                    encoded,
                    device="mixed" if cfg["device"] == "gpu" else "cpu",
                    start_frame=start[0],
                    sequence_length=cfg["seq"],
                )
                outs += [video, start]
            return tuple(outs)
        for r, spec in enumerate(specs):
            kw = dict(spec)
            if "filenames" in kw:
                kw["labels"] = list(range(len(kw["filenames"])))
            elif legacy:
                kw["file_list_frame_num"] = True  # frame numbers, not timestamps; end exclusive
            else:
                kw["file_list_format"] = "frames"
                kw["file_list_include_end"] = False  # explicit: the default changed over time
            if frame_num:
                kw["enable_frame_num"] = True
            outs += reader(
                device=cfg["device"],
                sequence_length=cfg["seq"],
                random_shuffle=cfg["shuffle"],
                prefetch_queue_depth=cfg["reader_depth"],
                name=f"Reader{r}",
                **kw,
            )
        return tuple(outs)

    return pipe()


def warm_up_cuda(device_id):
    """Initializes CUDA and the executor so build_s measures the reader, not the runtime."""
    import nvidia.dali.fn as fn
    from nvidia.dali.pipeline import pipeline_def

    @pipeline_def(batch_size=1, num_threads=1, device_id=device_id)
    def tiny():
        return fn.random.uniform(shape=[1]).gpu()

    p = tiny()
    p.build()
    p.run()


class DaliRunner:
    """The DALI reader. run() returns the (frames, height, width) of every sample."""

    def __init__(self, cfg, specs, args):
        import nvidia.dali as dali

        self.cfg = cfg
        self.version = (
            dali.__version__
            + {
                "dali": "",
                "dali_legacy": " (fn.readers.video)",
                "dali_decoder": " (fn.readers.file + video decoder)",
            }[cfg["backend"]]
        )
        # up to `depth` (pipeline queue) + `reader_depth` (the reader's own queue) batches
        # may be produced before the clock starts
        self.prefetched = cfg["depth"] + cfg["reader_depth"]
        warm_up_cuda(args.device_id)
        self.pipe = build_pipeline(cfg, specs, args.device_id, args.seed)

    def build(self):
        self.pipe.build()

    def run(self):
        out = self.pipe.run()
        # shapes are host-side metadata; video outputs are every other output (video, label)
        return [tuple(s[:3]) for r in range(self.cfg["readers"]) for s in out[2 * r].shape()]


def sample_list(specs, seq, shuffle, num_frames):
    """The samples the DALI reader produces for `specs`: (path, start, end) tuples.

    Whole files are split into full sequences of `seq` frames (the reader's default step
    and stride); a file list gives one sample per line. DALI shuffles with its own seed, so
    a shuffled list has the same samples in a different order.
    """
    (spec,) = specs
    samples = []
    if "filenames" in spec:
        for path in spec["filenames"]:
            n = num_frames(path)
            samples += [(path, s, s + seq) for s in range(0, n - seq + 1, seq)]
    else:
        with open(spec["file_list"]) as f:
            for line in f:
                path, _, start, end = line.split()
                samples.append((path, int(start), int(end)))
    if shuffle:
        random.Random(0).shuffle(samples)
    return samples


class ReferenceRunner:
    """PyNvVideoCodec or torchcodec, called directly through their own APIs.

    Each sample of a batch is decoded by one of `threads` threads; every thread keeps its
    own decoder while consecutive samples come from the same file. Outputs are what the
    libraries return; torch is used only to wait for the GPU.
    """

    prefetched = 0  # a batch is decoded when it is requested

    def __init__(self, cfg, specs, args):
        import itertools
        from concurrent.futures import ThreadPoolExecutor

        import torch

        self.cfg, self.torch = cfg, torch
        self.backend, self.gpu = cfg["backend"], cfg["device"] == "gpu"
        self.device_id = args.device_id
        if self.gpu:
            torch.cuda.set_device(self.device_id)
            torch.zeros(1, device="cuda")  # CUDA context, so build_s measures the decoders
            torch.cuda.synchronize()
        if self.backend == "pynvvideocodec":
            import PyNvVideoCodec as nvc

            self.nvc, self.version = nvc, f"PyNvVideoCodec {nvc.__version__}"
        else:
            import torchcodec

            self.version = f"torchcodec {torchcodec.__version__}"
        self.version += f", torch {torch.__version__}"
        self.local = threading.local()
        # torchcodec on the GPU: whether a decoder reported decoding on the CPU instead
        # (None: not known, e.g. older torchcodec without VideoDecoder.cpu_fallback)
        self.cpu_fallback = None
        if self.gpu and self.backend != "pynvvideocodec":
            from torchcodec.decoders import VideoDecoder

            self.cpu_fallback = False if hasattr(VideoDecoder, "cpu_fallback") else None
        samples = sample_list(specs, cfg["seq"], cfg["shuffle"], self._num_frames)
        self.samples = itertools.cycle(samples)
        # no more threads than samples per batch: an idle extra thread would take the next
        # batch's samples of the same file and make its decoder seek
        workers = min(cfg["threads"], cfg["batch"])
        self.pool = ThreadPoolExecutor(workers) if workers > 1 else None

    def _open(self, path):
        if self.backend == "pynvvideocodec":
            return self.nvc.SimpleDecoder(
                path,
                gpu_id=self.device_id,
                use_device_memory=True,
                output_color_type=self.nvc.OutputColorType.RGB,  # interleaved, as DALI
            )
        from torchcodec import decoders

        if not self.gpu:
            return decoders.VideoDecoder(path, device="cpu")
        # the CUDA backend is chosen explicitly, as the default changed: FFmpeg's NVDEC support
        # up to torchcodec 0.11 (the only one before 0.8), torchcodec's own NVDEC decoder later
        # ("nvdec", called "beta" before, which later versions still accept)
        if not hasattr(decoders, "set_cuda_backend"):
            return decoders.VideoDecoder(path, device=f"cuda:{self.device_id}")
        backend = "ffmpeg" if self.backend == "torchcodec" else "beta"
        with decoders.set_cuda_backend(backend):
            return decoders.VideoDecoder(path, device=f"cuda:{self.device_id}")

    def _num_frames(self, path):
        return len(self._decoder(path))

    def _decoder(self, path):
        """This thread's decoder, switched to `path`."""
        local = self.local
        if getattr(local, "path", None) == path:
            return local.decoder
        if self.gpu:
            self.torch.cuda.set_device(self.device_id)  # this thread's current device
        if self.backend == "pynvvideocodec" and getattr(local, "decoder", None) is not None:
            local.decoder.reconfigure_decoder(path)  # picks a cached NVDEC decoder if any
        else:
            local.decoder = self._open(path)
        local.path = path
        return local.decoder

    def _decode(self, sample):
        """Returns the sample's (frames, height, width) and the library's output."""
        path, start, end = sample
        decoder = self._decoder(path)
        # the libraries' own outputs, not converted or copied
        if self.backend == "pynvvideocodec":
            frames = decoder.get_batch_frames_by_index(list(range(start, end)))
            return (len(frames),) + tuple(frames[0].shape[:2]), frames  # frames of H, W, C
        video = decoder.get_frames_in_range(start, end).data
        if self.gpu and getattr(decoder, "cpu_fallback", None):  # torchcodec >= 0.11
            self.cpu_fallback = True
        return (video.shape[0], video.shape[2], video.shape[3]), video  # F, C, H, W

    def build(self):
        pass

    def run(self):
        # The outputs of a batch are kept until the next run(), as Pipeline.run() outputs, so
        # that the process memory includes a whole batch (DALI's also includes its queues)
        self.outputs = None
        batch = [next(self.samples) for _ in range(self.cfg["batch"])]
        decoded = list(
            self.pool.map(self._decode, batch) if self.pool else map(self._decode, batch)
        )
        if self.gpu:
            self.torch.cuda.synchronize()  # as Pipeline.run(): outputs are ready
        self.outputs = [out for _, out in decoded]
        return [shape for shape, _ in decoded]


def make_runner(cfg, specs, args):
    dali = cfg["backend"] in ("dali", "dali_legacy", "dali_decoder")
    return (DaliRunner if dali else ReferenceRunner)(cfg, specs, args)


def percentile(sorted_vals, q):
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    return statistics.quantiles(sorted_vals, n=100, method="inclusive")[q - 1]


def run_one(cfg, specs, args):
    """Runs one configuration in this process and returns its metrics."""
    mon = GpuMonitor(args.device_id)
    res = dict(pre=mon.snapshot())

    runner = make_runner(cfg, specs, args)  # initializes CUDA before the clock starts
    res["backend_version"] = runner.version
    t0 = time.perf_counter()  # build_s: the DALI reader (indexing), not the CUDA runtime
    runner.build()
    if isinstance(runner, DaliRunner):  # other libraries open their decoders lazily, in run()
        res["build_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    runner.run()
    res["first_batch_ms"] = (time.perf_counter() - t0) * 1e3
    res["shape"] = list(runner.run()[0])

    iters = cfg.get("iters", args.iters)
    for _ in range(args.warmup):
        runner.run()
    # Up to runner.prefetched batches (DALI: `depth` in the pipeline queue + `reader_depth`
    # in the reader's own queue) may have been produced before the clock started. They are
    # excluded, and --iters/--min_time apply to the remaining window.
    skip = runner.prefetched
    stamps, frames = [], []
    mon.start()
    t_start = time.perf_counter()
    while True:
        shapes = runner.run()  # actual (frames, height, width) of every sample
        stamps.append(time.perf_counter())
        frames.append(sum(s[0] for s in shapes))
        kept = len(stamps) - skip
        if kept > 0 and kept >= iters:
            t_ref = stamps[skip - 1] if skip else t_start
            if stamps[-1] - t_ref >= args.min_time:
                break
    res.update(mon.stop())
    span = stamps[-1] - t_ref
    n = len(stamps) - skip
    times_ms = sorted((b - a) * 1e3 for a, b in zip([t_ref] + stamps[skip:], stamps[skip:]))
    res.update(
        iters=n,
        iters_skipped=skip,
        frames_per_s=sum(frames[skip:]) / span,
        ms_per_sample=span * 1e3 / (n * cfg["readers"] * cfg["batch"]),
        iter_p50_ms=statistics.median(times_ms),
        iter_p90_ms=percentile(times_ms, 90),
        iter_max_ms=times_ms[-1],
    )
    if getattr(runner, "cpu_fallback", None) is not None:
        res["cpu_fallback"] = runner.cpu_fallback
    res["post"] = mon.snapshot()
    if mon.errors:  # after the last NVML query, so that every failure is recorded
        res["nvml_error"] = "; ".join(mon.errors)
    pre, post = res["pre"], res["post"]
    res["other_pids_seen"] = sorted(
        mon.others | set(pre.get("other_pids", [])) | set(post.get("other_pids", []))
    )
    res["contaminated"] = bool(res["other_pids_seen"] or pre.get("dev_nvdec_util", 0) > 5)
    return res


def hash_one(cfg, specs, args):
    """Returns one checksum per sample of the first batches (see HASH_MIN_SAMPLES and
    --hash_epochs)."""
    pipe = build_pipeline(cfg, specs, args.device_id, args.seed, frame_num=True)
    pipe.build()
    hashes = []
    # Not dependent on threads or reader_depth, so that --compare --ignore can match runs
    batches = -(-HASH_MIN_SAMPLES // (cfg["batch"] * cfg["readers"]))  # ceil
    if args.hash_epochs:
        epoch = max(pipe.epoch_size(f"Reader{r}") for r in range(cfg["readers"]))
        batches = max(batches, -(-args.hash_epochs * epoch // cfg["batch"]) + 1)
    for _ in range(batches):
        outs = [_as_cpu(t) for t in pipe.run()]
        for r in range(cfg["readers"]):
            video, label, frame_num = outs[3 * r : 3 * r + 3]
            for i in range(len(video)):
                h = hashlib.sha1(video.at(i).tobytes())
                h.update(label.at(i).tobytes())
                h.update(frame_num.at(i).tobytes())
                hashes.append(h.hexdigest())
    return dict(hashes=hashes, hash_batches=batches)


def _as_cpu(tensor_list):
    return tensor_list.as_cpu() if hasattr(tensor_list, "as_cpu") else tensor_list


def child_main(args):
    job = json.loads(args._single)
    cfg, specs = job["cfg"], job["specs"]
    try:
        res = hash_one(cfg, specs, args) if args.hash else run_one(cfg, specs, args)
    except Exception as e:
        lines = [ln for ln in str(e).strip().splitlines() if ln.strip()]
        detail = [ln for ln in lines if "rror" in ln or "ssert" in ln or "failed" in ln]
        res = dict(error=f"{type(e).__name__}: " + " | ".join((detail or lines)[-3:]))
    print("RESULT " + json.dumps(res))


def run_child(cfg, specs, args):
    cmd = [sys.executable, os.path.abspath(__file__), "--_single"]
    cmd += [json.dumps(dict(cfg=cfg, specs=specs))]
    for flag in ("dali_extra", "device_id", "iters", "warmup", "min_time", "seed", "hash_epochs"):
        cmd += [f"--{flag}", str(getattr(args, flag))]
    cmd += ["--hash"] if args.hash else []
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=args.timeout)
    except subprocess.TimeoutExpired:
        return dict(error=f"timeout after {args.timeout} s")
    for line in reversed(p.stdout.splitlines()):
        if line.startswith("RESULT "):
            return json.loads(line[len("RESULT ") :])
    tail = [ln for ln in (p.stderr or p.stdout).splitlines() if ln.strip()][-15:]
    detail = [ln for ln in tail if "rror" in ln or "ssert" in ln] or tail[-3:]
    return dict(error=f"exit code {p.returncode}: " + " | ".join(detail[-3:]))


def aggregate(cfg, reps):
    """Median and spread over repetitions."""
    ok = [r for r in reps if "error" not in r]
    res = dict(cfg)
    res["repeats"] = len(reps)
    if not ok:
        res["error"] = reps[-1]["error"] if reps else "no runs"
        return res
    res["errors"] = [r["error"] for r in reps if "error" in r]
    res["contaminated"] = any(r.get("contaminated") for r in ok)
    if any("cpu_fallback" in r for r in ok):
        res["cpu_fallback"] = any(r.get("cpu_fallback") for r in ok)
    res["runs"] = ok
    for key in (
        "frames_per_s",
        "ms_per_sample",
        "iter_p50_ms",
        "iter_p90_ms",
        "iter_max_ms",
        "build_s",
        "first_batch_ms",
        "proc_nvdec_util",
        "proc_mem_max_mb",
    ):
        vals = [r[key] for r in ok if r.get(key) is not None]
        if vals:
            res[key] = statistics.median(vals)
            if key in ("frames_per_s", "build_s", "first_batch_ms") and len(vals) > 1:
                res[key + "_spread_pct"] = (max(vals) - min(vals)) / res[key] * 100
    res["shape"] = ok[0].get("shape")
    for key in ("backend_version", "nvml_error"):
        if ok[0].get(key):
            res[key] = ok[0][key]
    return res


# ----------------------------------------------------------------------------------------
# reporting


COLUMNS = [
    ("scenario", "scenario", "{:<13}"),
    ("backend", "backend", "{:<16}"),
    ("files", "files", "{:<17}"),
    ("layout", "layout", "{:<12}"),
    ("device", "dev", "{:<3}"),
    ("batch", "bs", "{:>2}"),
    ("seq", "seq", "{:>3}"),
    ("threads", "thr", "{:>3}"),
    ("reader_depth", "rq", "{:>2}"),
    ("readers", "rd", "{:>2}"),
    ("shuffle", "sh", "{:<2}"),
    ("frames_per_s", "frames/s", "{:>8.1f}"),
    ("frames_per_s_spread_pct", "spr%", "{:>4.1f}"),
    ("ms_per_sample", "ms/smp", "{:>7.2f}"),
    ("iter_p50_ms", "p50 ms", "{:>8.2f}"),
    ("iter_p90_ms", "p90 ms", "{:>8.2f}"),
    ("iter_max_ms", "max ms", "{:>8.2f}"),
    ("build_s", "build s", "{:>7.3f}"),
    ("first_batch_ms", "1st ms", "{:>7.1f}"),
    ("proc_nvdec_util", "dec%", "{:>4.0f}"),
    ("proc_mem_max_mb", "mem MB", "{:>6.0f}"),
]


def _width(fmt):
    return int(fmt[3:].split(".")[0].rstrip("}"))


def print_header():
    print("  ".join(head.rjust(_width(fmt)) for _, head, fmt in COLUMNS) + "  flags")


def format_row(r):
    cells = []
    for key, head, fmt in COLUMNS:
        v = r.get(key)
        if key == "shuffle":
            v = "y" if v else "n"
        width = _width(fmt)
        try:
            cells.append(fmt.format(v) if v is not None else "-".rjust(width))
        except (ValueError, TypeError):
            cells.append(str(v).ljust(width))
    flags = []
    if r.get("contaminated"):
        flags.append("CONTAMINATED")
    if r.get("frames_per_s_spread_pct", 0) > 3:
        flags.append("spread>3%")
    if r.get("cpu_fallback"):
        flags.append("CPU fallback")  # reported by torchcodec (>= 0.11)
    if r.get("backend") in LEAD_IN_BACKENDS and r.get("files") in LEAD_IN_FILES:
        flags.append("lead-in frames")  # not the same frames as DALI, see LEAD_IN_FILES
    if r.get("errors"):
        flags.append(f"{len(r['errors'])} failed run(s)")
    if r.get("error"):
        flags.append("ERROR: " + r["error"][:160])
    return "  ".join(cells) + ("  " + ", ".join(flags) if flags else "")


def provenance(args):
    info = dict(
        time=time.strftime("%Y-%m-%d %H:%M:%S"),
        host=socket.gethostname(),
        python=sys.version.split()[0],
        cpu_count=os.cpu_count(),
        argv=sys.argv,
        env={k: v for k, v in os.environ.items() if k.startswith("DALI_") or k.startswith("CUDA_")},
    )
    try:
        import nvidia.dali as dali

        info["dali_version"] = dali.__version__
        info["dali_git_sha"] = getattr(dali, "__git_sha__", None)
        info["dali_path"] = os.path.dirname(dali.__file__)
    except Exception as e:
        info["dali_error"] = str(e)
    from importlib import metadata

    for pkg in ("PyNvVideoCodec", "torchcodec", "torch"):
        try:
            info.setdefault("packages", {})[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            continue  # optional: needed only by the --backends that use it
    try:
        nvml, h = nvml_handle(args.device_id)
    except Exception as e:
        info["nvml_error"] = str(e)
        return info
    queries = dict(
        gpu=lambda: nvml.nvmlDeviceGetName(h),
        gpu_uuid=lambda: nvml.nvmlDeviceGetUUID(h),
        driver=lambda: nvml.nvmlSystemGetDriverVersion(),
        other_gpu_pids=lambda: other_gpu_pids(nvml, h),
        nvdec_util_at_start=lambda: nvml.nvmlDeviceGetDecoderUtilization(h)[0],
    )
    for key, query in queries.items():
        try:  # one unsupported query must not hide the others
            value = query()
            info[key] = value.decode() if isinstance(value, bytes) else value
        except Exception as e:
            info.setdefault("nvml_query_errors", {})[key] = str(e)
    return info


def compare(a_path, b_path, ignore):
    """Compares per-sample hashes of matching configurations in two --hash files.

    Fields listed in `ignore` (e.g. threads, reader_depth) are left out when matching,
    so a configuration in A is compared against every configuration in B that differs
    only in those fields. The output must not depend on them.
    """
    with open(a_path) as fa, open(b_path) as fb:
        a, b = json.load(fa), json.load(fb)
    skip = {"hashes", "scenario", "error", "errors"} | set(ignore)

    def key(r):
        return json.dumps({k: v for k, v in r.items() if k not in skip}, sort_keys=True)

    def label(r):
        return ", ".join(f"{k}={r[k]}" for k in sorted(r) if k not in ("hashes", "errors", "error"))

    b_runs = {}
    for r in b["results"]:
        b_runs.setdefault(key(r), []).append(r)
    a_keys = {key(r) for r in a["results"]}
    bad = compared = 0
    # a configuration that failed or is missing on either side fails the comparison: a
    # build that cannot decode a workload must not pass as "identical"
    for r in b["results"]:
        if key(r) not in a_keys:
            bad += 1
            print(f"MISSING in {a_path}: {label(r)}")
    failed_runs = set()  # (file, index): each failed run is reported and counted once
    for path, results in ((a_path, a["results"]), (b_path, b["results"])):
        for i, r in enumerate(results):
            if "hashes" not in r and (path, i) not in failed_runs:
                failed_runs.add((path, i))
                bad += 1
                print(f"FAILED in {path}: {label(r)}  ({r.get('error', 'no checksums')})")
    for r in a["results"]:
        matches = b_runs.get(key(r), [])
        if not matches:
            bad += 1
            print(f"MISSING in {b_path}: {label(r)}")
        for other in matches:
            if "hashes" not in r or "hashes" not in other:
                continue  # reported above
            n = max(len(r["hashes"]), len(other["hashes"]))
            diff = sum(x != y for x, y in zip(r["hashes"], other["hashes"]))
            diff += abs(len(r["hashes"]) - len(other["hashes"]))
            compared += 1
            bad += diff > 0
            vs = ", ".join(f"{k}={other[k]}" for k in ignore if other.get(k) != r.get(k))
            status = "OK  " if diff == 0 else "DIFF"
            print(f"{status} {diff:4d}/{n} samples  {label(r)}" + (f"  vs {vs}" if vs else ""))
    if compared == 0 and bad == 0:
        print("nothing to compare: no matching configurations")
        return 2
    print("all identical" if bad == 0 else f"{bad} comparison(s) differ, failed or are missing")
    return 1 if bad else 0


# ----------------------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="fn.experimental.readers.video benchmark on DALI_extra data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("-s", "--scenarios", default="all", help="comma separated, or all")
    parser.add_argument(
        "-j", "--threads", default="1,2,4,8,16", help="num_threads sweep (B200: 1,2,4,7,8,14,16)"
    )
    parser.add_argument("--readers", default="1,2,4,8", help="reader counts for 'engines'")
    parser.add_argument(
        "-b",
        "--backends",
        default="dali",
        help=f"comma separated, any of {list(BACKEND_SCENARIOS)}; others for reference",
    )
    parser.add_argument("-d", "--device_id", default=0, type=int)
    parser.add_argument("--repeat", default=3, type=int, help="fresh-process runs per config")
    parser.add_argument("--iters", default=20, type=int, help="minimum timed iterations")
    parser.add_argument("--min_time", default=5.0, type=float, help="minimum timed seconds")
    parser.add_argument("--warmup", default=3, type=int, help="untimed iterations")
    parser.add_argument("--timeout", default=900, type=int, help="seconds per run")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--dali_extra", default=os.environ.get("DALI_EXTRA_PATH", ""))
    parser.add_argument("--json", help="write results (with provenance) to this file")
    parser.add_argument("--quick", action="store_true", help="smoke test: tiny sweep, 1 run")
    parser.add_argument("--hash", action="store_true", help="record per-sample checksums")
    parser.add_argument(
        "--hash_epochs",
        default=0,
        type=int,
        help="with --hash: checksum at least this many epochs of every configuration, and "
        "one more batch",
    )
    parser.add_argument("--compare", nargs=2, metavar=("A", "B"), help="compare two --hash files")
    parser.add_argument(
        "--ignore",
        default="",
        help="with --compare: comma separated config fields to ignore when matching runs, "
        "e.g. threads,reader_depth to check that output does not depend on them",
    )
    parser.add_argument("--_single", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.compare:
        return parser, args
    if not args.dali_extra:
        parser.error("--dali_extra is empty; set DALI_EXTRA_PATH or pass --dali_extra.")
    args.dali_extra = os.path.abspath(args.dali_extra)
    if args.hash and not args.json and not args._single:
        parser.error("--hash requires --json to store the checksums; got no --json.")
    args.backends = args.backends.split(",")
    unknown = [b for b in args.backends if b not in BACKEND_SCENARIOS]
    if unknown:
        parser.error(
            f"--backends got unknown backend(s) {unknown}; expected a subset of "
            f"{list(BACKEND_SCENARIOS)}."
        )
    if args.hash and args.backends != ["dali"]:
        parser.error(f"--hash supports only the dali backend; got --backends {args.backends}.")
    if args.quick:
        args.threads, args.readers, args.repeat = "1,4", "1,2", 1
        args.iters, args.min_time, args.warmup = 3, 0.0, 1
    return parser, args


def main():
    parser, args = parse_args()
    if args.compare:
        sys.exit(compare(*args.compare, [f for f in args.ignore.split(",") if f]))
    if args._single:
        child_main(args)
        return

    threads = [int(t) for t in args.threads.split(",")]
    readers = [int(r) for r in args.readers.split(",")]
    all_scenarios = scenarios(threads, readers, args.quick)
    if args.scenarios == "all":
        names = [n for n in all_scenarios if args.hash or n not in HASH_ONLY_SCENARIOS]
    else:
        names = args.scenarios.split(",")
    unknown = [n for n in names if n not in all_scenarios]
    if unknown:
        parser.error(
            f"--scenarios got unknown scenario(s) {unknown}; expected 'all' or a subset of "
            f"{list(all_scenarios)}."
        )
    hash_only = [n for n in names if n in HASH_ONLY_SCENARIOS]
    if hash_only and not args.hash:
        parser.error(f"--scenarios {hash_only} run only with --hash.")
    if args.hash:
        args.repeat = 1

    prov = provenance(args)
    print("# " + json.dumps(prov))
    if prov.get("nvml_error"):
        warnings.warn(f"NVML is unavailable ({prov['nvml_error']}); GPU checks are disabled.")
    busy = prov.get("other_gpu_pids") or (prov.get("nvdec_util_at_start") or 0) > 5
    if busy:
        details = []
        if prov.get("other_gpu_pids"):
            details.append(f"pids {prov['other_gpu_pids']}")
        if prov.get("nvdec_util_at_start") is not None:
            details.append(f"NVDEC utilization {prov['nvdec_util_at_start']}%")
        warnings.warn(
            f"The GPU is in use by other processes ({', '.join(details)}); "
            f"the results will be skewed."
        )

    workdir = tempfile.mkdtemp(prefix="video_reader_bench_")
    results, failed = [], 0
    try:
        jobs, not_run = [], set()
        for name in names:
            for base, backend in [(c, b) for c in all_scenarios[name] for b in args.backends]:
                if not backend_runs(backend, name, base):
                    not_run.add((backend, name, base["threads"]))
                    continue
                cfg = dict(base, scenario=name, backend=backend)
                try:
                    specs = materialize(cfg, args.dali_extra, workdir)
                except Exception as e:
                    jobs.append((cfg, None, f"{type(e).__name__}: {e}"))
                    continue
                jobs.append((cfg, specs, None))
        for backend, name, threads in sorted(not_run):
            print(f"# not run: {name} threads={threads} with {backend} (see backend_runs)")
        for cfg, specs, err in jobs:
            if specs:
                paths = []
                for s in specs:
                    if "filenames" in s:
                        paths += s["filenames"]
                    else:
                        with open(s["file_list"]) as f:
                            paths += [ln.split()[0] for ln in f]
                prewarm(os.path.realpath(p) for p in paths)
        reps = {i: [] for i in range(len(jobs))}
        rng = random.Random(args.seed)
        for rep in range(args.repeat):
            order = list(range(len(jobs)))
            if rep > 0:
                rng.shuffle(order)  # round-robin, so drift does not line up with a config
            for i in order:
                cfg, specs, err = jobs[i]
                res = dict(error=err) if err else run_child(cfg, specs, args)
                reps[i].append(res)
                note = res.get("error") or (
                    f"{res['frames_per_s']:.1f} frames/s" if "frames_per_s" in res else "done"
                )
                print(
                    f"[run {rep + 1}/{args.repeat}] {cfg['scenario']} {cfg['backend']} "
                    f"{cfg['files']} {cfg['layout']} bs={cfg['batch']} thr={cfg['threads']} "
                    f"rq={cfg['reader_depth']} rd={cfg['readers']}: {note}",
                    file=sys.stderr,
                    flush=True,
                )
        if not args.hash:
            print_header()
        for i, (cfg, specs, err) in enumerate(jobs):
            if args.hash:
                r = dict(cfg, seed=args.seed, **reps[i][0])  # with the hash_batches used
                n = len(r.get("hashes", []))
                print(
                    f"{cfg['scenario']} {cfg['files']} threads={cfg['threads']}: "
                    f"{n} sample hashes" + (f"  ERROR: {r['error']}" if r.get("error") else "")
                )
            else:
                r = aggregate(cfg, reps[i])
                print(format_row(r))
            # a configuration fails if any of its repetitions failed
            failed += bool(r.get("error") or r.get("errors"))
            results.append(r)
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    if args.json:
        with open(args.json, "w") as f:
            json.dump(dict(provenance=prov, results=results), f, indent=1)
        print(f"wrote {args.json}")
    if failed:
        print(f"{failed} configuration(s) failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
