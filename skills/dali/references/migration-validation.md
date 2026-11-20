# Migration validation

Run these checks before reporting a migration to DALI done.

## Preserve

- Sample-label association, ordering, shuffling, sharding, batching and checkpointing.
- Augmentation semantics and randomness, including per-sample versus per-batch random parameters.
- Dtype, layout, shape, channel order, value range and normalization.
- Interpolation and antialiasing.
- Last-batch handling and epoch length.

## Check outputs

- Run the old and new paths on the same fixed inputs.
- Compare deterministic stages numerically, with a tolerance.
- Where pixels cannot match exactly, compare a task metric instead, such as accuracy on a held-out batch or recognized text.
- Compare random stages by the distribution of their parameters, not bit for bit.

## Measure

Measure end-to-end throughput on the same hardware and settings, excluding warmup, before claiming any speedup.
