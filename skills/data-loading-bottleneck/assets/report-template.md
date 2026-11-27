# Data-loading result

| Decision | Result |
|---|---|
| **Verdict** | **[DETECTED / POTENTIAL / NOT DETECTED / INCONCLUSIVE]** |
| **Cause** | [Supported concrete stage or capacity limit, or `unresolved composite`] |
| **Next action** | [One evidence-backed recommendation, or the evidence needed to resolve the result] |

## Workload

[Command, model, settings, environment, dataset source, format, cardinality, and cache state]

**Substitutions**

| Change | Why it was required | Expected bias and verdict scope |
|---|---|---|
| [canonical -> measured] | [reason] | [effect and where the conclusion applies] |

## Detection

| Run | Full-step window | Throughput | Exposed loader wait |
|---|---:|---:|---:|
| Real | [value, invalid, or unavailable] | [value, invalid, or unavailable] | [value and percent, invalid, or unavailable] |
| Replay | [value, invalid, or unavailable] | [value, invalid, or unavailable] | [value and percent, invalid, or unavailable] |

**Result:** [Measured speedup, wait-only prediction from Real, their difference, and threshold
interpretation. If Replay is invalid or unavailable, report the primary effect]

**Replay boundary:** [What replay bypassed and what remained, or why it was unavailable]
 
**Distributed behavior**

| Rank | Samples | Full-step window | Exposed loader wait | Pre-barrier timestamp | Useful GPU / NCCL |
|---:|---:|---:|---:|---:|---:|
| [rank] | [value] | [value] | [value and percent] | [value] | [value / value or unavailable] |

**Rank result:** [Slowest-window aggregation, wait and arrival skew, and whether collectives
mask starvation]

## Localization

**Primary Nsight Systems report**

```text
/absolute/path/to/trace.nsys-rep
```

**Profile summary**

```text
/absolute/path/to/profile-summary.json
```

**Trace navigation**

| Domain | PID(s) | Range names |
|---|---|---|
| [domain] | [PID list] | [names] |

**GPU timeline**

- [Active and idle result]
- [Transfer or other critical-path result]

**End-to-end attribution**

| Stage (main / worker / GPU) | Status | P50 | Critical-path evidence | Artifact |
|---|---|---:|---|---|
| [stage] | [measured / absent / composite / unavailable] | [value, N, and unit] | [dependency, observed overlap and denominator, or why unavailable] | [summary or timeline] |

**Cause result:** [Supported actionable cause, unresolved composite and limit, or missing
evidence.]

## Recommendation

[Recommendation]

## Confidence and scope

- **Equivalence:** [Real <=> Replay: `pass` (`exact` or `work-equivalent`), `fail`, or
  `unavailable`. Real <=> Profile: give one of those statuses for each conclusion. Cite the
  relevant identities or batch signatures, steps, lifecycle, operating-regime evidence, and
  any restrictions]
- **Missing coverage:** [Main-process ranges or worker PIDs without ranges]
- **Profiler perturbation:** [Profiled versus unprofiled difference, or unavailable]
- **Limits:** [For INCONCLUSIVE, name the evidence needed]

**Artifacts**

| Absolute path | Status | Purpose / reason |
|---|---|---|
| [/path/to/primary-artifact] | primary | [detection or diagnosis] |
| [/path/to/artifact] | [supporting / restricted / topology only / superseded / invalid] | [purpose, restriction, or rejection reason] |
