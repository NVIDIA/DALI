# Profiling and Localization

## Select the production target

Profile the production command before isolating stages. Use the Nsight Systems result from
preflight and test the `nvtx` import. If needed, make one project-appropriate installation
attempt. If either tool remains unavailable, return to `SKILL.md` before instrumenting.

For distributed training, choose the target from per-rank Real timing. Use rank 0 when
exposed waits are balanced. Otherwise, profile the rank with the largest exposed wait and its
workers. Per-rank timing carries skew, while `summarize_nsys.py` reads one report. An
NCCL-blocked rank may only show a starving peer's symptom. Keep ranges rank-local and
aggregate samples with the slowest-rank window.

Select the rank with a launcher wrapper:

```bash
#!/bin/sh
# torchrun --no-python --nproc_per_node=<N> /bin/sh ./profile_rank.sh <production-python> train.py ...
production_python=${1:?production Python required}
shift
if [ "${RANK:?torchrun did not set RANK}" = "${PROFILE_RANK:-0}" ]; then
  exec nsys profile --output="${PROFILE_OUTPUT:-data-loading-rank-${RANK}}" <options> "$production_python" "$@"
fi
exec "$production_python" "$@"
```

Retain the command, revision or diagnostic diff, profiler version, summary, and artifact
paths. Allow one capture and at most one recapture. Spend the recapture on an unusable
capture, excessive overhead, or one dominant composite. Then report any remaining limit.

## Add semantic ranges

Use manual annotations in the domain `data-loading`. Bare gaps in an unlabeled GPU trace
conflate loader waits, synchronization, and scheduler noise. Add these main-process ranges:

- `capture_session`: outer capture activation before loader iteration and warmup
- `profile_window`: synchronized measurement boundary after warmup
- `batch_wait`: each blocking loader retrieval
- `train_step`: full consumer step exposed by the integration boundary
- `forward` and `backward`: separate compute phases. Use `compute` only when separation is
  impossible and report that limitation

Add `device_transfer`, `optimizer`, or synchronization when useful. Name boundaries
precisely. A callback after device transfer, for example, must not imply that `train_step`
includes transfer.

```python
import nvtx

DOMAIN = "data-loading"

with nvtx.annotate("batch_wait", domain=DOMAIN):
    batch = next(iterator)

with nvtx.annotate("train_step", domain=DOMAIN):
    with nvtx.annotate("forward", domain=DOMAIN):
        loss = model(batch)
    with nvtx.annotate("backward", domain=DOMAIN):
        loss.backward()
    optimizer.step()
```

`forward` and `backward` measure host-side launch of asynchronous CUDA work, not GPU
execution. The device usually synchronizes later, at loss readback or the step boundary.
Take GPU compute time from the CUDA activity in the profile summary, not from these range
widths.

For managed loops, place equivalent ranges at the loader wrapper, batch callbacks,
production forward, and backward hooks. Use `nvtx.start_range`/`end_range` across
callbacks, and close every handle from a `finally` path.

### Semantic colors

Apply this Solarized mapping consistently to main and worker NVTX ranges. Annotate every
material workload-specific operation needed for attribution, even if it is absent from the
table, and use the nearest semantic color. Colors encode broad categories. Range names
identify exact operations.

| Meaning | Color |
|---|---|
| Structural parents | `base00` `#657b83` |
| Source open/read | `cyan` `#2aa198` |
| Decode/conversion | `violet` `#6c71c4` |
| Preprocessing | `orange` `#cb4b16` |
| Batch assembly | `green` `#859900` |
| Loader wait | `yellow` `#b58900` |
| Compute | `blue` `#268bd2` |
| Device transfer | `red` `#dc322f` |
| Synchronization | `magenta` `#d33682` |

Pass the colors as integers, not strings, to NVTX. For instance: `color=0x6c71c4`.

## Mark workers

With active workers, annotate the material stages present, such as source access,
materialization or parsing, per-sample processing, batch assembly, and handoff. Choose
boundaries from the workload code before capture. Split safely separable composites that
prevent attribution. Generic labels such as fetch, transform, and preprocess cannot support
a cause when their material children can be annotated safely. Refine a dominant composite
only if the recommendation could change. Otherwise, leave it unsplit and record the limit.
Image decode, augmentation, tokenization, and patchify are examples rather than requirements.

Place ranges where work executes. Lazy APIs can move cost. Pillow `Image.open` parses
metadata but normally defers pixel decoding to `load()`, `convert()`, or pixel access. End
work ranges before `yield`, `await`, or blocking queue operations, and attribute asynchronous
work from its execution events. If instrumentation forces earlier materialization, confirm
that it preserves consumer-visible output, work, ordering, memory behavior, and relevant
timing.

Annotate concrete operations rather than one range around the whole item:

```python
import io
import nvtx
from PIL import Image
from torch.utils.data import Dataset

WORKER_DOMAIN = "data-loading-worker"

class ExampleDataset(Dataset):
    def __getitem__(self, index):
        with nvtx.annotate("read_file", domain=WORKER_DOMAIN, color=0x2AA198):
            encoded = self.paths[index].read_bytes()
        with nvtx.annotate("decode_jpeg", domain=WORKER_DOMAIN, color=0x6C71C4):
            image = Image.open(io.BytesIO(encoded))
            image.load()
        with nvtx.annotate("resize", domain=WORKER_DOMAIN, color=0xCB4B16):
            image = self.resize(image)
        with nvtx.annotate("random_crop", domain=WORKER_DOMAIN, color=0xCB4B16):
            image = self.crop(image)
        with nvtx.annotate("to_tensor", domain=WORKER_DOMAIN, color=0xCB4B16):
            tensor = self.to_tensor(image)
        with nvtx.annotate("normalize", domain=WORKER_DOMAIN, color=0xCB4B16):
            return self.normalize(tensor)
```

Each range names one operation a recommendation can act on. A range around the entire
transform pipeline may dominate while identifying nothing removable. For an opaque
`transforms.Compose`, annotate each entry in its `transforms` list instead of the outer call.

`image.load()` forces the decode inside its own range. Without it, Pillow defers that cost
into the next one.

When the dataset cannot be edited, wrap it and split the wrapped range into concrete stages
before naming a cause. Preserve all loader behavior:

- Define wrapper classes and functions at module scope for spawn and forkserver. This applies
  only to definitions, not constructed `nvtx.annotate`, domain, or registered-message objects.
  See the fork rule below.
- Preserve batched `Dataset.__getitems__`. Wrapping only `__getitem__` may disable a fast
  path, and defining `__getitems__` on a dataset that lacks one forces the batched path
  production never runs, because PyTorch selects it with `hasattr`.
- Wrap, do not replace, custom collation.
- Forward required attributes explicitly. Generic `__getattr__` can recurse during worker
  reconstruction.

Prefer direct stage annotations if wrappers would affect type checks, sharding, sampling,
batched fetch, or lifecycle. If worker instrumentation is unsafe or impossible, state why.
Main-process waits cannot identify worker sub-stages.

### Register annotations in workers

`nvtx.get_domain` and `Domain.get_registered_string` cache process-owned handles with
`lru_cache`. A forked worker inherits its ancestor's cached handles, so constructing the
annotation after the fork can still reuse an invalid handle. This also applies to a forkserver
that imported the module. Spawn starts a fresh interpreter and is unaffected.

The failure is silent. The message appears in `StringIds` without child ranges, errors, or a
dropped-event signal, and an unmatched pop can corrupt neighboring counts. Keep only wrapper
definitions, domains, and message strings at module scope. Construct annotations inside the
worker without registering their pair in an ancestor. If that is impossible, use a
worker-only pair first registered in the child or report worker-stage coverage as
unavailable. Leave the caches intact.

## Capture with Nsight Systems

```bash
nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none --cpuctxsw=process-tree \
  --capture-range=nvtx \
  --nvtx-capture=capture_session@data-loading \
  --capture-range-end=stop \
  --force-overwrite=true \
  --output=<artifact-dir>/data-loading \
  <production-python> train.py ...
```

Add `--trace-fork-before-exec=true` only when worker-stage attribution requires following
active fork or forkserver workers. Omit it for `num_workers == 0` and spawn. This option can
substantially perturb, crash, or deadlock the target, especially alongside other process-tree
tracing. Omitting it for a fork-based loader makes worker-stage coverage unavailable.

Start `capture_session` immediately before creating the profiled loader iterator so
activation cannot pause the consumer at the measurement boundary and fill the prefetch queue.
Warm inside the session, enclose the synchronized measurement in `profile_window`, and end
`capture_session` afterward. If a managed loop prevents activation before iterator startup,
activate early, discard the transient, and warm through at least one full prefetch cycle
before `profile_window`. An empty capture means the configured range name or domain never
fired. If `--cpuctxsw=process-tree` is unsupported, use `none` and do not infer worker CPU
pressure from scheduling.

Keep `--capture-range-end=stop`: it ends collection after `capture_session` while allowing
training to continue. Nsight otherwise defaults to `stop-shutdown`, which terminates the
target. Close both structural ranges normally. The summarizer rejects an open
`profile_window`.

Use manual ranges in this process-tree capture. Do not add `--python-functions-trace`.
Python tracing plus pre-exec worker following can materially perturb fork/forkserver loaders.

If NVTX capture triggering fails, spend the recapture on one fallback. Prefer retaining the
outer range and triggering with the CUDA Profiler API:

```python
torch.cuda.profiler.start()
try:
    with nvtx.annotate("capture_session", domain="data-loading"):
        # create the iterator and warm up
        with nvtx.annotate("profile_window", domain="data-loading"):
            # measured work
            pass
finally:
    torch.cuda.profiler.stop()
```

Replace `--capture-range=nvtx` with `--capture-range=cudaProfilerApi`. Retain
`--capture-range-end=stop`. If that trigger is unavailable, use a time-bounded capture
instead. Do not try both.

## Summarize the measured window

After a detailed capture, extract clipped range and CUDA statistics without loading raw
profiler dumps into context:

```bash
<production-python> <skill-dir>/scripts/summarize_nsys.py \
  <artifact-dir>/data-loading.nsys-rep \
  --output <artifact-dir>/profile-summary.json
```

Use the JSON for range counts and distributions, worker PIDs, stored colors, GPU activity,
CUDA copies, and loader-wait/GPU-idle overlap. The summary establishes neither hierarchy nor
causality, so inspect the timeline before assigning a critical path, cause, or recommendation.
`batch_wait_overlapping_gpu_idle_percent` uses loader wait as its denominator.
`gpu_idle_overlapping_batch_wait_percent` uses GPU idle.

## Validate the capture and overhead

Use the profile summary and timeline to confirm:

- one `capture_session` encloses warmup and measurement
- one `profile_window` encloses the measurement
- main-process range counts match the bounded window on every captured rank
- every worker expected to be captured has worker ranges
- capture extends beyond the prefetched queue. Otherwise, the trace shows queue drain, not
  production behavior.

An empty worker PID list with a limitation about ranges that "could not be named" means
coverage is indeterminate. Those ranges remain in the trace but the summary could not label
them. Missing worker PIDs without that limitation are a missing-marker failure unless worker
following was deliberately omitted.

Missing required markers or invalid structural ranges make a capture unusable. Spend the
recapture if available, or report the gap. Worker ranges overlap, so never sum their
durations. Correlate them with queue consumption and main-process `batch_wait`.

Restrict statistics to `profile_window` and state every overlap or utilization denominator.
Range `sum_ns` is an invocation sum rather than elapsed time. Never add nested parent and
child ranges, parallel workers, or separate CUDA category unions. Compute GPU active time
from the union of kernel, copy, and memset intervals. Use CUDA copy events, not host-range
width, to attribute transfers.

Compare the profiled window directly with the retained unprofiled Real baseline. After
warmup, profiling must preserve the amount and path of measured full-step work and its
operating regime. Match batch count and work-affecting signatures, completed steps, producer
lifecycle, routing and topology, synchronization, and update behavior. Different records,
random values, resulting numerical state, or exact stall positions are acceptable unless
they change the work or critical-path relationship being reported. When content, order, or
locality can change a claim, compare the relevant workload or record identities. Use
identifiers when event correlation requires them.

Accept profiler perturbation only when throughput stays within ~10% and no systematic stall
or lifecycle pattern appears. For a larger difference, use the recapture, if available. Keep
`cuda`, `nvtx`, manual ranges, and `--sample=none`. Drop `osrt` and set `--cpuctxsw=none`.
Retain `--trace-fork-before-exec=true` only for worker-stage attribution. Without it, limit
conclusions to the main process and GPU. After recapture, apply a work or operating-regime
mismatch only to the conclusions it could affect. Preserve conclusions supported by
independent equivalent work.

## Interpret

Build one end-to-end hierarchy from the workload code and trace. Cover the report template's
stages, using their names as guidance rather than an allow-list.

### Attribute and stop

Begin with the observed loss of full-step progress. Correlate `batch_wait` with GPU-idle time,
device transfers with delayed model work, and producer stages with delivered batches. Trace
backward with batch or record identifiers when timing alone is insufficient. Report
wall-clock overlap with its denominator and reserve range sums for per-invocation service
demand. Name a limiting stage only when the timeline or measured capacity shows that it
materially gated batch delivery or model work.

Use starvation as detection evidence only. Establish the cause separately. A dominant unsplit
stage remains `unresolved composite`, even when its aggregate demand predicts throughput.

A call stack, blocked-thread state, scheduler event, or overlap identifies the wait site. It
does not quantify removable causal cost. Report `blocked at <site>` until the mechanism is
supported.

Name an inferred mechanism such as a lock, GIL, allocator, IPC, or storage as the cause only
when measured service demand and available capacity predict observed throughput while
excluding competing paths. A controlled one-factor intervention that consistently improves
equivalent full-step windows provides the same support. Until then, label it
`candidate: <mechanism>` and state the cheapest falsification test. Stop once a validated
trace identifies a concrete critical stage with sufficient evidence. A dominant stack frame
alone is insufficient.

- Use OS/runtime events as corroboration for semantic ranges, not as a substitute.
- Continuously scheduled workers suggest CPU or memory/IPC pressure. Low scheduled time
  plus long reads suggests storage. Queue waits can mean idleness or backpressure.
- Attribute transfer from GPU copy events and whether they delay model work, not
  host-range width.
- In DDP, compare per-rank exposed waits and useful non-NCCL GPU work before blaming
  collectives. Boundary arrival skew may stay small because ranks reconverge inside the
  window.
- Compare equivalent kernel work before claiming input contention slowed compute.

Map the measured critical path to one recommendation. Do not privilege worker scaling:

- **Source access dominates:** recommend the indicated storage investigation or change,
  such as latency, bandwidth, cache, format, or local staging.
- **Decode or preprocessing dominates:** recommend optimizing or eliminating the hot CPU
  operation. When several consecutive compatible stages dominate and GPU headroom exists,
  consider a contiguous accelerator pipeline from encoded input through model-ready GPU
  tensors. Note operation and variable-shape support, batching, layout, dtype, randomness,
  semantic equivalence, and GPU contention. Avoid partial offload that adds host-device
  round trips. DALI replay proves bypass headroom, not production-pipeline feasibility.
- **Batch assembly or IPC dominates:** recommend examining payload bytes, early dtype
  inflation, copies, shared memory, or packing.
- **Workers stay busy and the work is safely parallelizable:** recommend a bounded
  `num_workers` experiment only when CPU, memory, and IPC headroom make it plausible.
- **Device transfer is critical:** recommend pinning or overlap based on GPU copy events.

## Conditional cases

- **Iterable/custom producer:** mark producer and queue boundaries.
- **Asynchronous/device preprocessing:** synchronize measurement boundaries and record
  whether replay contains host or device data.
- **Remote storage:** mark the client/read path. Local file events may miss
  network-library work.
- **Variable batches:** compare samples or bytes and verify shape distributions.
