<!-- Always include the decision table, Workload, Detection, and `Confidence and scope`.
DETECTED/POTENTIAL require Cause and Next action. Add Localization when profiled and
Recommendation when supported. Without profiling, set Cause to `unresolved composite`. For NOT
DETECTED, omit Localization and Recommendation. Cause and Next action do not apply. For
INCONCLUSIVE, set Cause to `unresolved composite`, name missing evidence under Next action,
and add Localization only when profiled. Put excluded input costs in Detection with value
and frequency, never in Cause. Keep Artifacts under Confidence, omit empty sections, and
print full `.nsys-rep` paths. -->

# Data-loading result

| Decision | Result |
|---|---|
| **Verdict** | **[DETECTED / POTENTIAL / NOT DETECTED / INCONCLUSIVE]** |
| **Cause** | [Supported concrete stage or capacity limit, or `unresolved composite`] |
| **Next action** | [One evidence-backed recommendation, or the evidence needed to resolve the result] |

## Workload

[Command, model, settings, environment, and dataset source, format, cardinality, and cache
state. Use one paragraph if the canonical command ran as-is.]

<!-- OPTIONAL: only when the measured run differed from canonical. -->
**Substitutions**

| Change | Why it was required | Expected bias and verdict scope |
|---|---|---|
| [canonical -> measured] | [reason] | [effect and where the conclusion applies] |

## Detection

<!-- Write `invalid` or `unavailable` for either run when no valid measurement exists. -->
| Run | Full-step window | Throughput | Exposed loader wait |
|---|---:|---:|---:|
| Real | [value, invalid, or unavailable] | [value, invalid, or unavailable] | [value and percent, invalid, or unavailable] |
| Replay | [value, invalid, or unavailable] | [value, invalid, or unavailable] | [value and percent, invalid, or unavailable] |

**Result:** [Measured speedup, wait-only prediction from Real, their difference, and threshold
interpretation. If Replay is invalid or unavailable, report the primary effect]

**Replay boundary:** [What replay bypassed and what remained, or why it was unavailable]
 
<!-- OPTIONAL for world_size > 1. Use Real timing for every rank. Include profiled GPU or
NCCL values only for captured ranks. -->
**Distributed behavior**

| Rank | Samples | Full-step window | Exposed loader wait | Pre-barrier timestamp | Useful GPU / NCCL |
|---:|---:|---:|---:|---:|---:|
| [rank] | [value] | [value] | [value and percent] | [value] | [value / value or unavailable] |

**Rank result:** [Slowest-window aggregation, wait and arrival skew, and whether collectives
mask starvation]

<!-- OPTIONAL: only when profiling ran. -->
## Localization

**Primary Nsight Systems report**

```text
/absolute/path/to/trace.nsys-rep
```

**Profile summary**

```text
/absolute/path/to/profile-summary.json
```

<!-- OPTIONAL for world_size > 1, where PIDs must be mapped to ranks. -->
**Trace navigation**

| Domain | PID(s) | Range names |
|---|---|---|
| [domain] | [PID list] | [names] |

**GPU timeline**

- [Active and idle result]
- [Transfer or other critical-path result]

<!-- Follow the actual source-to-training path with one row per profile-summary range group.
Include applicable source/read, parsing or materialization, per-record children,
batching/packing, handoff/IPC, exposed wait, device copies, forward/loss, backward, optimizer,
synchronization, and residual time. Indent children and mark every row measured, absent,
composite, or unavailable. Parallel or nested sums are not wall-time contribution. -->
**End-to-end attribution**

| Stage (main / worker / GPU) | Status | P50 | Critical-path evidence | Artifact |
|---|---|---:|---|---|
| [stage] | [measured / absent / composite / unavailable] | [value, N, and unit] | [dependency, observed overlap and denominator, or why unavailable] | [summary or timeline] |

**Cause result:** [Supported actionable cause, unresolved composite and limit, or missing
evidence.]

<!-- OPTIONAL: only when localization supports an optimization. -->
## Recommendation

<!-- State the recommendation, motivating measurement, expected mechanism, feasibility
constraint, and `untested` status. -->

## Confidence and scope

<!-- Only what could change the interpretation. Omit a bullet with nothing to report. -->

- **Equivalence:** [Real <=> Replay: `pass` (`exact` or `work-equivalent`), `fail`, or
  `unavailable`. Real <=> Profile: give one of those statuses for each conclusion. Cite the
  relevant identities or batch signatures, steps, lifecycle, operating-regime evidence, and
  any restrictions]
- **Missing coverage:** [Main-process ranges or worker PIDs without ranges]
- **Profiler perturbation:** [Profiled versus unprofiled difference, or unavailable]
- **Limits:** [For INCONCLUSIVE, name the evidence needed]

**Artifacts**

<!-- Include invalid and superseded runs with their rejection reasons. -->

| Absolute path | Status | Purpose / reason |
|---|---|---|
| [/path/to/primary-artifact] | primary | [detection or diagnosis] |
| [/path/to/artifact] | [supporting / restricted / topology only / superseded / invalid] | [purpose, restriction, or rejection reason] |
