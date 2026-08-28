---
name: data-loading-bottleneck
description: "Diagnose input-bound PyTorch training. Use for low or bursty GPU utilization, slow batches, num_workers tuning, preprocessing regressions, or input stalls. Not for model/kernel optimization."
license: Apache-2.0
compatibility: Requires Linux, Python, Git, an NVIDIA GPU and driver, and CUDA-enabled PyTorch.
permissions:
  - file_read
  - file_write
  - shell
  - network
  - env
metadata:
  author: "DALI Team <dali-team@nvidia.com>"
  tags:
    - pytorch
    - training
    - performance
    - data-loading
    - profiling
    - dali
  languages:
    - python
  team: dali
  domain: deep-learning
  version: "1.0.0"
---

# Data Loading Bottleneck

## Purpose

Determine whether PyTorch training is input-bound and, if so, localize one cause without
changing the production path, lifecycle, data flow, or topology.

## Prerequisites

Requires Linux, Git, CUDA, and PyTorch. DALI replay and Nsight/NVTX profiling are optional.
Preflight attempts their allowed setup and routes around anything unavailable.

## Instructions

### 1. Preflight and route

Create an artifact directory for commands and raw output, then run preflight with the
production Python:

```bash
<production-python> <skill-dir>/scripts/collect_preflight.py \
  --source-dir <checkout> --artifact-dir <artifact-dir> \
  --data-path <local-dataset> \
  [--expected-visible-gpus N]
```

For remote or custom input, use `--data-source <description>` instead of `--data-path`.
Preflight records only the description, while the production run validates access. Stop on a
hard blocker. If a sandbox hides the accelerator, rerun preflight and GPU work in production.
CPU execution is not a substitute. Record any environment change made in response to a
warning.

Start with the replay-support result from preflight. If replay is unavailable, use
`torch.version.cuda` to choose one package for a single isolated installation attempt:

- CUDA 12.x: `nvidia-dali-cuda120`
- CUDA 13.x: `nvidia-dali-cuda130`
- Other or unknown: record the unsupported runtime and skip replay.

```bash
<production-python> -m pip install --target <artifact-dir>/dali-deps <dali-package>
```

After a successful installation, append the target with `site.addsitedir()` and retry
`from nvidia.dali.plugin.pytorch.loader_evaluator import LoaderEvaluator`. Keep
target-installed dependencies behind production packages. If the import succeeds, use the
same setup for Real and Replay. If it fails, preserve the failure, leave production unchanged,
and profile after Real. Do not try another installation strategy. Use the production Python
and worktree `PYTHONPATH` for every run.

After preflight, record the original checkout status and diff, then create a disposable Git
worktree. Reproduce the canonical code and configuration, including staged, unstaged, and
relevant untracked changes. Put its package root or `src` directory first on `PYTHONPATH`,
point out-of-tree builds to it, and record representative module `__file__` paths. Stop if it
cannot reproduce the workload. Leave the original checkout untouched. Remove the worktree
after diagnosis and keep the artifacts.

### 2. Instrument one bounded production run

Instrument the production training path in the disposable worktree. Use a standalone harness
only when that path cannot be bounded or instrumented, as described in Troubleshooting.
Locate the last blocking loader retrieval at the intended replay boundary and the complete
optimizer update that consumes its batch. Record device transfer relative to the boundary,
batch/sample accounting, distributed topology, and implicit defaults. When
`LoaderEvaluator` is available, read `references/pytorch-dali.md` and build its paired Real
and Replay loaders. Without it, use the bounded production loader directly.

Set `prefetch_depth` to the batches that can be ready at the replay boundary, including
production wrapper buffers. Per rank, it must be at least `num_workers * prefetch_factor`.
Use 2 when `prefetch_factor` is unset and one when `num_workers == 0`. Choose
`measured_batches > prefetch_depth`. Set warmup and drain to at least that depth, then set
`total_batches = warmup_batches + measured_batches + drain_batches`. Bound the source to
that total, warm the first part, time the measured window, and drain outside it.

Keep the same workload, integration boundary, and timed window through localization. Add
only semantic ranges. Preserve work-affecting batch
structure, routing, topology, synchronization, and update behavior. Time complete steps and
every blocking loader retrieval. Synchronize the device and count samples at both window
boundaries. With multiple ranks, add boundary barriers and record each rank. Include normal
wait variability. In Real and Profile, do not manipulate the page cache or replace production
input with synthetic, repeated, modified, or deliberately pre-cached data.

If instrumentation fails, follow Troubleshooting. If no valid Real window remains after
those attempts, report `INCONCLUSIVE` and continue at §6.

Runs with material changes to the data source, sampling rules, batching, preprocessing, or
work-affecting input distribution are substitutes and cannot support canonical
`NOT DETECTED`.

### 3. Run Real

In a fresh process, run the bounded window through `LoaderEvaluator(mode="log")` when
available. Otherwise, use the bounded production loader. Per rank, record measured
samples, complete-step time, exposed loader wait, and the timestamp immediately before each
window barrier. Classify the source cache state as known warm before Real, warmed only by
this run's normal access, or unknown.

```text
aggregate throughput = sum(samples across ranks) / max(rank window duration)
```

Do not infer balanced ranks from final arrival skew alone. Collectives can repeatedly
reconverge imbalanced ranks. Continue at §4 when `LoaderEvaluator` is available. Otherwise,
continue at §5.

### 4. Run Replay and classify

Start another fresh process with the same `LoaderEvaluator` wrapper in `replay` mode. Change
only the mode. Any other change invalidates the comparison. Apply the post-boundary work
equivalence and lifecycle checks in `references/pytorch-dali.md` before classification.

If either run emits fewer than `total_batches`, report `INCONCLUSIVE` and continue at §6.
For any other unavailable or invalid Replay, continue at §5.

For a valid comparison:

```text
speedup = replay aggregate throughput / real aggregate throughput
```

| speedup | Verdict |
|---|---|
| `>1.50x` | `DETECTED` |
| `>1.10x` and `<=1.50x` | `POTENTIAL` |
| `<=1.10x` | `NOT DETECTED` |

These fixed heuristics follow DALI's [Data Loading Bottleneck Detection tutorial](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/frameworks/pytorch/loader_evaluator/pytorch_data_loader_evaluator.html).
They are not estimates of run-to-run noise.

Do not repeat Real or Replay. Continue at §6 after `NOT DETECTED`, and at §5 after
`DETECTED` or `POTENTIAL`.

### 5. Profile and localize

When §3 or §4 routes here, read `references/profiling.md`. If Nsight Systems, NVTX, or profile
summarization is still unavailable after its allowed setup, skip capture. Keep a valid Replay
verdict and mark localization unavailable. Without valid Replay, report `INCONCLUSIVE`.
Continue at §6.

Otherwise, reuse a valid Real run as the unprofiled baseline. Follow the reference's
single-capture workflow and shared limit of one recapture for any reason.

Apply the reference's structural and Real-baseline checks to each conclusion. A structurally
invalid capture is unusable. A work mismatch restricts only the conclusions it could affect,
while a valid Replay verdict remains authoritative. Without valid Replay, report `DETECTED`
only when validated profile evidence shows an input-path stage materially delaying full-step
progress in the measured window. Otherwise, report `INCONCLUSIVE`.

Use that delay as detection evidence. Attribute a cause only to a measured concrete stage or
supported capacity limit. Leave an unsplit limiting stage as `unresolved composite`. When
localization supports an optimization, give one ranked, evidence-backed recommendation,
mark it untested, and do not implement or benchmark it. Otherwise, state the missing evidence.

### 6. Report

Use `assets/report-template.md` and return the completed report in the final response. Do not
replace it with a path to a Markdown file. Include the sections and tables required for the
outcome, cite the evidence for the verdict and any recommendation, and scope the result to
the measured workload and recorded source cache state.

## Available Scripts

| Script | Purpose | Arguments |
|---|---|---|
| `scripts/collect_preflight.py` | Record environment, source, data, and optional-tool readiness in `preflight.json` | `--source-dir`, `--artifact-dir`, one of `--data-path` or `--data-source`; optional `--expected-visible-gpus` |
| `scripts/summarize_nsys.py` | Summarize the measured NVTX window, CUDA activity, and loader-wait/GPU-idle overlap | input `.nsys-rep` and required `--output` JSON path |

Both scripts write to the requested path, print status to stdout, and send diagnostics to
stderr. Preflight returns `0` when ready, `1` when blocked, and `2` on an operational or
usage error. The summarizer returns `0` on success, `1` for an invalid report or output, and
`2` for invalid arguments. Where a `run_script` helper exists, use it with the same
repo-relative script and arguments. Otherwise, use the documented production-Python commands.

## Examples

- “Find out whether data loading is causing bursty GPU utilization in this PyTorch training
  job, and identify the limiting stage.”
- “Optimize this model's attention kernels” is outside this skill's scope.

## Limitations

- Covers steady-state CUDA-enabled PyTorch training. Inference, startup, epoch transitions,
  checkpointing, and offline data preparation are outside the workflow.
- Determines whether the input path limits full-step training throughput. It does not
  optimize model, kernel, optimizer, or communication performance.
- Produces workload-specific conclusions. It does not predict behavior under another
  topology, data path, or cache state.

## Troubleshooting

- **Instrumented run failure:** rerun the canonical command without instrumentation and save
  both commands and outputs to separate workload failure from instrumentation failure.
- **Resource failure:** make at most one nearest runnable attempt that changes one resource
  setting. Label it a substitute and restrict the verdict accordingly.
- **Production path cannot be bounded or instrumented:** save the failed production attempt
  before using a standalone harness. Treat the harness as a substitute. State what differs or
  is missing and limit every conclusion that depends on it. Without the complete training
  step, it cannot establish that production is input-bound.
