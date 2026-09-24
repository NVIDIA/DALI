# DALI Loader Evaluation

## Build paired loaders

Use identically bounded production sources and the same `LoaderEvaluator` configuration.
Apply the bound at loader construction or the production entrypoint:

```python
import itertools
import torch

class BoundedDataLoader(torch.utils.data.DataLoader):
    def __init__(self, max_batches, **kwargs):
        super().__init__(**kwargs)
        self.max_batches = max_batches

    def __iter__(self):
        return itertools.islice(super().__iter__(), self.max_batches)

    def __len__(self):
        try:
            return min(self.max_batches, super().__len__())
        except TypeError:
            return self.max_batches

total_batches = warmup_batches + measured_batches + drain_batches
bounded = BoundedDataLoader(total_batches, dataset=dataset, **production_loader_kwargs)
cache_batches = len(bounded)
source = LoaderEvaluator(
    bounded,
    mode="replay" if replay_mode else "log",
    num_cached_batches=cache_batches,
)
```

Use `mode="log"` for Real and `mode="replay"` for Replay. Construction exhausts the bounded
source because `num_cached_batches` limits retention rather than consumption. Exclude
construction from timing. An OOM or unbounded construction is invalid.

`LoaderEvaluator` requires a finite `len()`, though the dataset may remain unsized. The
adapter returns `max_batches`, while Real's emitted count still detects early exhaustion.
Replay emits only the declared length, so validate retained cache count and post-boundary
work separately.

In both modes, warm `warmup_batches`, time `measured_batches`, then consume `drain_batches`
outside the window. The drain keeps source exhaustion and worker shutdown out of timing.

Keep production loader kwargs, dataset, sampler, batch sampler, sharding, and `set_epoch`
unchanged. Before Replay, estimate retained memory from the largest Real boundary batch and
configured cache size. Account for every local rank when batches use shared host memory and
every device when they remain on GPU. If the cache clearly cannot coexist with training
state, record the estimate, skip Replay, and profile.

Cache the whole bound when practical. A smaller cache cycles batches and is valid only when
the retained batches represent Real's timed post-boundary work and their objects are safe to
reuse. Mutations or aliasing may fail only on wraparound. Require
`0 < cache_batches <= len(bounded)`. Record configured and retained counts and whether Replay
cycles them.

When a framework type-checks or re-instantiates loaders, use a framework-specific
`DataLoader` shim. Delegate iteration to the evaluator and preserve required production
attributes such as `dataset`, `sampler`, and the bound. Demonstrate equivalent behavior. For
Lightning DDP, disable automatic sampler replacement and validate each rank's lifecycle.

## Prove equivalent timed work

After warmup, Real and Replay are equivalent when cached substitution preserves the amount
and path of timed work after the replay boundary. Preserve batch structure, routing, topology,
synchronization, update behavior, and safe object use. Different records, random values, or
resulting numerical state are acceptable unless they change that work.

Record sample counts and the work-affecting boundary signature: container and tensor
structure, shapes, dtypes, and bytes. For variable batches, record actual samples or bytes
and compare the shape sequence or distribution required by the consumer. Stable record keys
help when values or order affect work and when correlating stalls. Identity is supporting
evidence rather than a universal requirement.

When exact order matters, use a dedicated sampler generator for map-style sources. For
iterable and framework-owned sources, use their seed or epoch controls and stable keys. Keep
the sampler generator separate from model randomness. An ordered shard list may substitute
for keys only when it determines the consumed content.

Inspect the consumer from the replay boundary through the complete update for value- or
order-dependent branches, routing, sparsity, compilation, skipped updates, and stateful
caches. Match their controlling inputs or show unchanged execution. Keep the model,
optimizer, precision, initial state, steps, and synchronization fixed. Ordinary numerical
divergence is acceptable when timed work stays the same.

For multiple ranks, cache the bounded source independently per rank while preserving
rank-local sharding and `set_epoch`. Confirm producers are quiescent during replay. Inspect
persistent-worker PIDs rather than assuming.

Cache construction may advance RNG, epoch, or library state. Let the required warmup absorb
ordinary transients. Reject only persistent changes that survive warmup and alter timed work
or consumer resources beyond Replay's intended removal of producer work and contention.

## Reject invalid Replay

If either run emits fewer than `total_batches`, report `INCONCLUSIVE`. Reject Replay for:

- unequal bounds, samples, or steps, or a cache that does not represent the timed
  post-boundary work;
- a material difference in batch structure, routing, topology, synchronization, update
  behavior, or other work that remains after warmup;
- mutated or aliased cached objects, unsafe asynchronous reuse, or cycling that changes the
  timed work; or
- an unexpected producer or framework lifecycle change beyond Replay's intentional producer
  quiescence.

Do not place the emitted-count check after the last `yield` inside `__iter__`: a consumer
that stops at the bound never resumes the generator. Ordinary wait variability does not
invalidate Replay.

## Interpret replay

Compare measured speedup with the wait-only prediction `1 / (1 - w)`, where `w` is exposed
wait fraction. Use it as a reference rather than a limit. Replay may also free worker CPU
capacity, while a measured loader call may include retained work such as device copies in a
blocking prefetch wrapper. Report measured and expected speedup and their difference.
Agreement checks consistency but does not prove equivalent work.

State what the cache bypasses and what remains. Report speedup as diagnostic headroom, never
as a promised gain. Prefetch can make queue drain look faster than worker supply.
