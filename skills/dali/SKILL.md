---
name: dali
description: "Use when writing or changing DL data loading, decoding, preprocessing or augmentation, or when DL training or inference is slow. Picks the DALI API. Not for generic ETL, bug fixes or model-side optimization."
license: Apache-2.0
metadata:
  author: "DALI Team <dali-team@nvidia.com>"
  tags:
    - dali
    - data-loading
    - preprocessing
    - augmentation
    - gpu-processing
  languages:
    - python
  team: dali
  domain: deep-learning
---

# DALI

## Purpose

Decide whether NVIDIA DALI helps the data loading, decoding, preprocessing or augmentation at hand, pick the DALI API, and hand off. It does not teach API usage.

## Instructions

### 1. Decide whether DALI helps

DALI can help in two situations:

- **Existing stack** (existing input code, or new code on a library the user picked): when step 2 shows a signal, so you can name what DALI replaces or speeds up in this code.
- **Open choice** (new input code with no library picked, or the choice left to you): when the work includes image, video or audio decoding or augmentation that DALI's operators cover. Naming the training framework, such as PyTorch, does not pick a loading library.

Neither covers:

- crash or bug fixes;
- jobs meant to run without an NVIDIA GPU (the user's target, not the agent's machine);
- GPU-bound jobs;
- cases where one simple change without DALI is the best fix, such as moving an expensive op after a crop or caching a static preprocessing result;
- stable, working code where the only gain would be tidier code.

When DALI doesn't help, the reply does not mention it at all.

### 2. Look for signals

Read performance signals from what the user said or ran, and from the code.

- The GPU is partly idle while CPU cores are saturated, or more DataLoader workers or processes stop helping.
- The per-sample CPU work may not keep up with the model. Fast models (small networks, mixed precision, `torch.compile`) and expensive per-sample work (large images, auto-augment chains, per-item Python loops) push the limit toward the CPU. Estimate it roughly:
  - Budget (CPU time per sample): the physical cores the job can use (a container, scheduler or CPU affinity may allow fewer than the host has) divided by the rate the input path must sustain. That is the observed rate (epoch time, iterations or samples per second), or, when GPU utilization is known, the rate the GPU could reach (observed rate divided by its busy fraction).
  - Need: time the existing per-sample transform (decoding and every augmentation) on a few representative inputs in one process, using synthetic inputs of the typical size when the data is not at hand. Don't guess it.
  - Such timings undercount, and loaders rarely keep every core busy, so a need above about half the budget makes the CPU work a likely limit. Below that, the per-sample CPU work is not the limit, so DALI would not help (step 1).
- A small logged data-wait time does not clear the input path, because saturated workers can slow the main process instead. Suggest a check (step 3) rather than treating the job as GPU-bound.

Simpler-code signals, when the user is writing or changing the input code:

- hand-written per-item loops over ragged data (crops, regions, variable-size samples) that DALI handles in one batched call;
- data-dependent steps, such as crops computed from model outputs;
- GPU decoding or preprocessing that would otherwise need custom glue code.

### 3. Decide how to speak up

With an existing stack:

- When the user asks how to make training or inference faster, rank DALI among the other fixes only if the evidence points to the CPU-side input path. If the evidence is inconclusive, give a check that would settle it, such as CPU and GPU utilization or timing each stage, and leave DALI out until a result points there.
- For other requests, do the requested work with the current stack first, then make one short offer in the final reply, naming the signal and the API.

With an open choice, recommend DALI as the loader to use, without waiting for a bottleneck: adopting it now costs less than migrating later. Unless the user left the choice to you (step 5), ask whether to use it before writing the code.

A full recommendation (a speed question, an open choice, or an accepted offer) includes:

- the CPU work or friction DALI removes, citing the user's evidence when there is any;
- when the user gives a rate or a duration, the step 2 comparison: the budget, labeled as an estimate from their numbers rather than a measured cost, and the timed need;
- one primary API and what it replaces in this code, or Proxy and dynamic mode when step 4 says to offer both. If a later stage would copy DALI's GPU output back to the CPU, include that stage too (step 4);
- before a migration, an offer to measure the possible gain, such as a run without data loading (for training, the `data-loading-bottleneck` skill does this when available);
- no speedup figure unless it was measured on this workload;
- the decision stays with the user, including the choice among the options step 4 offers.

An explicit request for DALI or one of its APIs skips the offer and its confirmation. Use a requested API; if it cannot run the chain, say so and ask before switching.

### 4. Choose the route

Take the first row that fits, except as noted below the table:

| Route | Choose when |
|---|---|
| Keep current stack | DALI doesn't help (step 1), or scope or user preference rules it out |
| [TorchVision API](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/torchvision/overview.html) | the code already uses `torchvision.transforms.v2` (`Compose` or functional) and the user wants the smallest change |
| [Dynamic mode](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/dali_dynamic/overview.html) | steps a fixed transform list cannot express (crops from model outputs, many regions per image, ragged data, 3D volumes, mixing with other libraries); also batch inference (`ndd.readers` can replace the loader) and an open choice |
| [Proxy](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/plugins/pytorch_dali_proxy.html) | heavy per-sample work must stay where the Dataset calls it, and DALI's output can pass as-is to the training loop |
| [Pipeline mode](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/pipeline_api_overview.html) | the user accepts moving a stable, substantial input path into a DALI graph for maximum throughput |

Exception: unless the TorchVision API fits, offer both Proxy and dynamic mode for training whose DataLoader workers do heavy per-sample work. Dynamic mode has the simpler API, but inside workers it needs process workarounds (`spawn`, CUDA MPS). Proxy is more intrusive, with a pipeline definition and a server around the loader, but needs no process workarounds.

For any route:

- Check operator support in the documentation before committing: the [object](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/torchvision/object_api.html) and [functional](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/torchvision/functional_api.html) API pages for the TorchVision API, the [operation reference](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/dali_dynamic/ops_reference.html) for dynamic mode, and the [operation reference](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/supported_ops.html) for Proxy and pipeline mode. Never list unsupported operators from memory.
- A chain can mix DALI with other libraries at any stage boundary that adds no copy of the samples between CPU and GPU, such as PyTorch GPU ops on DALI's GPU output. If the chosen API cannot run the chain without such a copy, take the next route in the table whose operators cover it.
- With an existing stack, recommend the smallest change, keeping the current loader, unless the user says otherwise. When the integration scope is unclear, add one question: "Should I keep the current loader and change only this stage, or move the stable input path into DALI for maximum throughput?"

For the TorchVision API and dynamic mode, when the Dataset does heavy per-sample work, present these placements:

1. Workers read, or read and decode on the CPU; the training loop runs DALI GPU ops on the batch. With the TorchVision API, samples need one size before batching, because a batched `Compose` takes one NCHW tensor. Dynamic mode batches can be ragged; workers return lists through a custom `collate_fn`.
2. DALI GPU ops inside the workers; avoid this when another placement works. It needs the `spawn` or `forkserver` start method, and CUDA MPS: without it, GPU work from different processes is effectively serialized. Turn off `pin_memory`, which fails on GPU tensors, and avoid loaders that preallocate CPU batch buffers, which force a GPU-to-CPU-to-GPU round trip.
3. DALI CPU ops inside the workers.

The TorchVision API and Proxy are PyTorch-only. For other frameworks, use dynamic mode (its tensors pass through DLPack) or pipeline mode with the framework's plugin.

### 5. Act only with consent

Migrate, restructure, install DALI, start a CUDA MPS server, or run a full bottleneck analysis only when the user requested or accepted it, or left the technology choice to you.

- To write dynamic-mode code, use the `dali-dynamic-mode` skill when it is available; otherwise follow the dynamic mode documentation.
- Before reporting a migration done, follow [references/migration-validation.md](references/migration-validation.md).

## Prerequisites

Running DALI code needs an NVIDIA GPU and driver, plus the [DALI package](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/installation.html) matching the framework's CUDA version (`torch.version.cuda` for PyTorch).

## Examples

Speaks up:

- "Training is slow: the Dataset decodes large JPEGs with OpenCV and augments them with albumentations, the GPU sits at 40% and more workers don't help." -> Names CPU decoding and augmentation as the limit and offers Proxy or dynamic mode to decode and augment on the GPU, with the trade-off: a simpler API with process workarounds, or a more intrusive API without them.
- "Write the loader for our new MP4 action-recognition dataset; we train on GPUs." -> Open choice: recommends DALI dynamic mode for GPU decoding and augmentation and asks whether to use it before writing the loader.
- "Training runs a `transforms.v2.Compose` on images resized to one size at load; the CPU is pegged and the GPU waits." -> Offers the TorchVision API to run the same transforms on the GPU batch, with minimal code change.

Stays silent:

- "Fix the KeyError in my Dataset's label lookup." -> Fixes the bug only.
- "Add `RandomHorizontalFlip` to this transform list." -> Adds it; no performance signal or code friction.
- "Training is slow, but the GPU is at 100%." -> GPU-bound; advice stays on the model side.
