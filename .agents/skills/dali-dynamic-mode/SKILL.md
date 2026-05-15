---
name: dali-dynamic-mode
description: "Use when writing DALI data loading or preprocessing code with `nvidia.dali.experimental.dynamic` (ndd), or when converting DALI pipeline-mode code to dynamic mode, or when the user asks about DALI dynamic mode, imperative DALI, or ndd. Use this skill any time someone mentions 'ndd', 'dynamic mode', or wants to load/augment data with DALI outside of a pipeline definition."
---

# DALI Dynamic Mode

Dynamic mode is DALI's imperative Python API. Call DALI operators as regular Python functions with standard control flow -- no pipeline graph, no `pipe.build()`, no `pipe.run()`.

```python
import nvidia.dali.experimental.dynamic as ndd
```

## Core Data Types

### Tensor -- single sample

```python
t = ndd.tensor(data)           # copy
t = ndd.as_tensor(data)        # wrap, no copy if possible
t.cpu()                        # move to CPU
t.gpu()                        # move to GPU
t.torch(copy=False)            # conversion to PyTorch tensor with no copy (default)
t[1:3]                         # slicing supported
np.asarray(t)                  # NumPy via __array__ (CPU only)
```

Supports `__dlpack__`, `__cuda_array_interface__`, `__array__`, arithmetic operators.

### Batch -- collection of samples (variable shapes OK)

```python
b = ndd.batch([arr1, arr2])    # copy
b = ndd.as_batch(data)         # wrap, no copy if possible
```

**Batch has no `__getitem__`** -- `batch[i]` raises `TypeError` because indexing is ambiguous (sample selection vs. per-sample slicing). Use the explicit APIs instead:

| Intent | Method | Returns |
|--------|--------|---------|
| Get sample i | `batch.select(i)` | `Tensor` |
| Get subset of samples | `batch.select(slice_or_list)` | `Batch` |
| Slice within each sample | `batch.slice[...]` | `Batch` (same batch_size) |

`.select()` picks **which samples**. `.slice` indexes **inside each sample**.

```python
xy = ndd.random.uniform(batch_size=16, range=[0, 1], shape=2)
crop_x = xy.slice[0]       # Batch of 16 scalars, first element from each sample
crop_y = xy.slice[1]       # Batch of 16 scalars, second element from each sample
sample_0 = xy.select(0)    # Tensor, the entire first sample [x, y]
```

**PyTorch conversion:**
- `batch.torch()` -- works for uniform shapes; raises for ragged batches
- `batch.torch(pad=True)` -- zero-pads ragged batches to max shape (use for variable-length audio, detection boxes, etc.)
- `batch.torch(copy=None)` is the default (avoids copy if possible)
- Batch has **no `__dlpack__`** -- use `ndd.as_tensor(batch)` first for DLPack consumers. `ndd.as_tensor` supports `pad` as well.
- `Tensor.torch(copy=False)` is default (no copy)

**Iteration:** `for sample in batch:` yields Tensors.

## Readers

Readers are **stateful objects** -- create once, reuse across epochs. This matters because readers track internal state like shuffle order and shard position.

```python
reader = ndd.readers.File(file_root=image_dir, random_shuffle=True)

for epoch in range(num_epochs):
    for jpegs, labels in reader.next_epoch(batch_size=64):
        # jpegs, labels are Batch objects
        ...
```

Key points:
- Reader outputs (jpegs, labels, etc.) are **CPU** tensors/batches. Labels typically stay on CPU until you convert them for your framework (e.g. `labels.torch().to(device)`).
- Reader classes are **PascalCase**: `ndd.readers.File(...)`, `ndd.readers.COCO(...)`, `ndd.readers.TFRecord(...)`
- `batch_size` goes to `next_epoch()`, not to the reader constructor
- `next_epoch(batch_size=N)` yields tuples of `Batch`; `next_epoch()` without batch_size yields tuples of `Tensor`
- The iterator from `next_epoch()` must be fully consumed before calling `next_epoch()` again
- Once a reader is used with a given batch_size, it cannot be changed. Similarly, a reader used in batch mode cannot switch to sample mode or vice versa.

Sharded reading for distributed training:
```python
reader = ndd.readers.File(
    file_root=image_dir,
    shard_id=rank, num_shards=world_size,
    stick_to_shard=True,
    pad_last_batch=True,
)
```

## Device Handling

- Device is **inferred from inputs** -- GPU if any input is on GPU
- For hybrid decode: use `device="gpu"` (NOT `"mixed"`). The `"mixed"` keyword is a pipeline-mode concept for implicit CPU-to-GPU transfer; in dynamic mode, passing `device="gpu"` triggers the same hardware-accelerated decode path.
- Don't call `.cpu()` before passing to a GPU model -- `.torch()` gives you a GPU tensor directly. `.cpu()` is only needed for consumers requiring host memory (numpy, `__array__`).
- CUDA stream sync between DALI and PyTorch is **automatic via DLPack** -- no manual stream management needed.

## Execution Model

Default mode is `eager` -- async execution in a background thread, returns immediately.

**No `.evaluate()` needed in most cases.** Any data consumption (`.torch()`, `__dlpack__`, `__array__`, `.shape`, property access, iteration) triggers evaluation automatically.

For debugging, switch to synchronous mode so errors surface at the exact call site rather than later in the async queue:

```python
with ndd.EvalMode.sync_full:
    images = ndd.decoders.image(jpegs, device="gpu")
    images = ndd.resize(images, size=[224, 224])
    # Any error surfaces here, at the exact op that failed
```

Modes (increasing synchronicity): `deferred` < `eager` < `sync_cpu` < `sync_full`

Use `EvalMode.sync_full` for debugging instead of scattering `.evaluate()` calls -- it's cleaner and catches all issues at once. `sync_cpu` is often sufficient and lighter than `sync_full`.

## Thread Configuration

```python
ndd.set_num_threads(4)  # Call once at startup, only if necessary to override the defaults
```

Controls DALI's internal worker threads for CPU operators. Defaults to CPU affinity count or `DALI_NUM_THREADS` env var. Unrelated to Python-level threading.

## RNG

Two approaches (use one, not both):

```python
# Approach 1: set the thread-local default seed (simple, good enough for most cases)
ndd.random.set_seed(42)
angles = ndd.random.uniform(batch_size=64, range=(-30, 30))

# Approach 2: explicit RNG object (finer control, pass rng= to each op)
rng = ndd.random.RNG(seed=42)
values = ndd.random.uniform(batch_size=64, range=[0, 1], shape=2, rng=rng)
```

When `rng=` is passed to a random op, the explicit RNG overrides the default seed. Thread-local: each thread has independent random state.

Random ops need an explicit `batch_size` when working with batches -- there is no pipeline-level batch size to inherit.

## Example: Image Classification Pipeline

```python
import nvidia.dali.experimental.dynamic as ndd

ndd.set_num_threads(4)
reader = ndd.readers.File(file_root="/data/imagenet/train", random_shuffle=True)

for epoch in range(num_epochs):
    for jpegs, labels in reader.next_epoch(batch_size=64):
        images = ndd.decoders.image(jpegs, device="gpu")
        images = ndd.resize(images, size=[224, 224])
        images = ndd.crop_mirror_normalize(
            images,
            mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
            std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        )
        train_step(images.torch(), labels.torch())
```

## Common Mistakes

| Wrong | Right | Why |
|-------|-------|-----|
| `device="mixed"` | `device="gpu"` | `"mixed"` is pipeline mode only |
| `batch[i]` | `batch.select(i)` | `Batch` has no `__getitem__` |
| `batch.select(0)` for per-sample slicing | `batch.slice[0]` | `.select()` picks samples; `.slice` slices within each sample |
| `.evaluate()` after every op | Let consumption trigger eval | `.torch()`, `.shape`, etc. trigger it automatically |
| `.cpu()` before GPU model | `.torch()` directly | Avoids wasteful D2H + H2D round-trip |
| Recreate reader each epoch | `reader.next_epoch()` | Readers are stateful -- create once, reuse |
| `ndd.readers.file(...)` | `ndd.readers.File(...)` | Reader classes are PascalCase |
| `break` from `next_epoch()` loop | Exhaust iterator or create new reader | Iterator must be fully consumed before next `next_epoch()` |
| No `batch_size` to random ops | `ndd.random.uniform(batch_size=N, ...)` | No pipeline-level batch size to inherit |

## Pipeline Mode Migration

| Pipeline Mode | Dynamic Mode |
|--------------|--------------|
| `@pipeline_def` / `pipe.build()` / `pipe.run()` | Direct function calls in a loop |
| `fn.readers.file(...)` | `ndd.readers.File(...)` (PascalCase, stateful) |
| `fn.decoders.image(jpegs, device="mixed")` | `ndd.decoders.image(jpegs, device="gpu")` |
| `fn.op_name(...)` | `ndd.op_name(...)` |
| Pipeline-level `batch_size=64` | `reader.next_epoch(batch_size=64)` + random ops `batch_size=64` |
| Pipeline-level `seed=42` | `ndd.random.set_seed(42)` or `ndd.random.RNG(seed=42)` |
| Pipeline-level `num_threads=4` | `ndd.set_num_threads(4)` at startup |
| `output.at(i)` | `batch.select(i)` |
| `output.as_cpu()` | `batch.cpu()` |
| `pipe.run()` returns tuple of `TensorList` | `reader.next_epoch(batch_size=N)` yields tuples of `Batch` |
