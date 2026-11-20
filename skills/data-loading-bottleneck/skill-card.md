## Description: <br>
Diagnose input-bound PyTorch training. Use for low or bursty GPU utilization, slow batches, num_workers tuning, preprocessing regressions, or input stalls. Not for model/kernel optimization. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and engineers training models with CUDA-enabled PyTorch use this skill to determine whether training is input-bound (via DALI LoaderEvaluator Real/Replay comparison and optional Nsight Systems profiling) and to localize one data-loading cause with an evidence-backed, untested recommendation. <br>

### Deployment Geography for Use: <br>
Global <br>

## Requirements / Dependencies: <br>
**Requires API Key or External Credential:** [Not Specified] <br>
**Credential Type(s):** [None identified] <br>

Do not include secrets in prompts/logs/output; use least-privilege credentials; rotate keys as appropriate. <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [profiling.md](references/profiling.md) <br>
- [pytorch-dali.md](references/pytorch-dali.md) <br>
- [DALI Data Loading Bottleneck Detection tutorial](https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/frameworks/pytorch/loader_evaluator/pytorch_data_loader_evaluator.html) <br>


## Skill Output: <br>
**Output Type(s):** [Analysis, Shell commands, Files] <br>
**Output Format:** [Markdown diagnostic report (from assets/report-template.md) with JSON artifacts (preflight.json, Nsight summary)] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Verdict is one of DETECTED, POTENTIAL, NOT DETECTED, or INCONCLUSIVE; recommendations are marked untested.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`) <br>
- Codex (`openai/openai/gpt-6.1-sol`) <br>



## Evaluation Tasks: <br>
3 evaluation tasks (2 positive, 1 negative): inference-latency, mobilenet-input-bound, and vit-not-input-bound; 1 attempt per task, each in its own isolated sandbox pod (`k8s-sandbox` environment), each compared against a no-skill baseline. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Is it safe to use? Scored from `security` (100%). <br>
- Correctness: Is the answer correct? Scored from `accuracy` (100%). <br>
- Discoverability: Was the right skill loaded when needed? Scored from `skill_execution` (100%). <br>
- Effectiveness: Did the skill help complete the task? Scored from `goal_accuracy` (50%) + `behavior_check` (50%). <br>
- Efficiency: Did it avoid wasted tool calls and token usage? Scored from `skill_efficiency` (50%) + `token_efficiency` (50%). <br>

Underlying evaluation signals used in this run: <br>
- `security`: Security: unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Skill Execution: whether the expected skill was selected, decoys were avoided, and the workflow executed. <br>
- `skill_efficiency`: Tool Productivity: tool-call productivity (legacy wire id; routing is scored under Discoverability). <br>
- `accuracy`: Accuracy: final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Goal Accuracy: whether the user's goal was achieved. <br>
- `behavior_check`: Behavior Check: whether the expected workflow behavior was followed. <br>
- `token_efficiency`: Token Efficiency: actual uncached prompt plus completion usage (50% of Efficiency). <br>



## Evaluation Results: <br>
**Overall verdict: PASS — Recommended for publication**

**Publication Recommendation:** Recommended for publication based on the completed evaluation evidence in this report.

### Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 83.2% — baseline ran, but no comparable score was available; uplift unavailable | 68.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 66.7% (-33.3 points) | 33.3% → 0.0% (-33.3 points) |
| Correctness | 73.3% → 80.0% (+6.7 points) | 73.3% → 100.0% (+26.7 points) |
| Discoverability | 90.5% — baseline ran, but no comparable score was available; uplift unavailable | 77.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 46.4% → 97.5% (+51.1 points) | 48.1% → 97.2% (+49.1 points) |
| Efficiency | 81.5% — baseline ran, but no comparable score was available; uplift unavailable | 68.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 88.3% → 75.4% (-12.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 75.6% → 73.4% (-2.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

### Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 8 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 3 task(s) |

## Skill Version(s): <br>
1.0.0 (source: frontmatter) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
