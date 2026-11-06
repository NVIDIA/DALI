# Skill Benchmark: data-loading-bottleneck

> ✅ **Overall verdict: PASS — Recommended for publication**

## Publication Recommendation

Recommended for publication based on the completed evaluation evidence in this report.

## Evaluation Metadata

- Skill: `data-loading-bottleneck`
- Validation status: `passed`
- Evaluation date: 2026-11-06
- Evaluator version: `1.6.9`
- Agents: Claude Code (`aws/anthropic/bedrock-claude-opus-5-5`), Codex (`openai/openai/gpt-6.1-sol`)
- Tasks: 3 evaluation tasks (2 positive, 1 negative)
- Dataset digest: `sha256:642f36038384748bef0620dd1c092a9fa4c44666923d949d47335b34db0df052` (skill-evaluator-dataset-snapshot/1)
- Attempts per task: 1
- Environment: `k8s-sandbox`
- Tier 2 evidence: required for publication
- Tier 3 evidence: required for publication

Each task attempt ran in its own isolated sandbox pod.

## Tier 1 LLM Provenance

SkillSpector provider, model, analyzer revision, and determinism evidence retained for this scan.

| Skill | Provider | Adapter | Analyzer | Model | Revision | Determinism |
| --- | --- | --- | --- | --- | --- | --- |
| data-loading-bottleneck | nv_inference | openai | mcp_tool_poisoning | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| data-loading-bottleneck | nv_inference | openai | semantic_developer_intent | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| data-loading-bottleneck | nv_inference | openai | semantic_quality_policy | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| data-loading-bottleneck | nv_inference | openai | semantic_security_discovery | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |
| data-loading-bottleneck | nv_inference | openai | meta_analyzer | `aws/anthropic/bedrock-claude-opus-5-5` | `2.13.4` | nondeterministic / provider_defaults |

- Controls for `data-loading-bottleneck`: temperature=not set (source=provider_default, forwarded=no); seed=not set (source=unset, forwarded=no); reasoning_effort=not set (source=provider_default, forwarded=no)

## What This Report Answers

The three-tier evaluation checks whether the skill:

- is safe to use;
- produces correct answers;
- is discovered and activated when needed;
- helps the agent complete the user's goal and expected workflow; and
- avoids wasted skill and tool usage.

## Results at a Glance

| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |
|---|---:|---:|
| Overall | 83.2% — baseline ran, but no comparable score was available; uplift unavailable | 68.7% — baseline ran, but no comparable score was available; uplift unavailable |
| Security | 100.0% → 66.7% (-33.3 points) | 33.3% → 0.0% (-33.3 points) |
| Correctness | 73.3% → 80.0% (+6.7 points) | 73.3% → 100.0% (+26.7 points) |
| Discoverability | 90.5% — baseline ran, but no comparable score was available; uplift unavailable | 77.5% — baseline ran, but no comparable score was available; uplift unavailable |
| Effectiveness | 46.4% → 97.5% (+51.1 points) | 48.1% → 97.2% (+49.1 points) |
| Efficiency | 81.5% — baseline ran, but no comparable score was available; uplift unavailable | 68.7% — baseline ran, but no comparable score was available; uplift unavailable |

**How to read this table:** baseline is the same task attempted without the target skill. Scores are rounded to one decimal; threshold-adjacent values use additional precision so their displayed band matches the verdict. Uplift is derived from those displayed scores and shown in percentage points.

Example: `47.0% → 92.0% (+45.0 points)` means the skill-assisted run scored 92.0%, 45.0 percentage points above its 47.0% no-skill baseline.

A partial dimension was calculated from only the available configured signals; review the detailed report before relying on it.

### Comparison details

Shared-source comparisons use the same signals and weights on both conditions. They do not replace the full dimension scores above or change the publication verdict.

| Agent | Dimension | Basis | Baseline → With skill (uplift) | Explanation |
| --- | --- | --- | --- | --- |
| claude-code | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| claude-code | Efficiency | Token-only comparison | 88.3% → 75.4% (-12.9 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |
| codex | Discoverability | Dimension comparison | N/A | Discoverability is not applicable to the no-skill baseline; there is no installed skill to discover. |
| codex | Efficiency | Token-only comparison | 75.6% → 73.4% (-2.2 points) | Only Token Efficiency is comparable; the full dimension scores use different evaluator sources. |

## Token Usage

Actual Tier 3 execution usage is reported for every observed agent/case pair and both conditions.

| Agent | Dataset case | With skill | Without skill | Delta | Change | Coverage |
|---|---|---:|---:|---:|---:|---|
| claude-code | All cases | 3,490,394 | 1,099,742 | +2,390,652 | +217.38% | skill 3/3; base 3/3 |
| claude-code | inference-latency | 520,995 | 362,539 | +158,456 | +43.71% | skill 1/1; base 1/1 |
| claude-code | mobilenet-input-bound | 1,737,249 | 320,551 | +1,416,698 | +441.96% | skill 1/1; base 1/1 |
| claude-code | vit-not-input-bound | 1,232,150 | 416,652 | +815,498 | +195.73% | skill 1/1; base 1/1 |
| codex | All cases | 3,971,839 | 3,368,526 | +603,313 | +17.91% | skill 3/3; base 3/3 |
| codex | inference-latency | 355,236 | 741,222 | -385,986 | -52.07% | skill 1/1; base 1/1 |
| codex | mobilenet-input-bound | 2,567,439 | 1,641,636 | +925,803 | +56.40% | skill 1/1; base 1/1 |
| codex | vit-not-input-bound | 1,049,164 | 985,668 | +63,496 | +6.44% | skill 1/1; base 1/1 |
| ALL AGENTS | Dataset aggregate | 7,462,233 | 4,468,268 | +2,993,965 | +67.01% | skill 6/6; base 6/6 |

Prompt tokens include cached reads, so total tokens are `prompt + completion` (cached is not added twice). The Efficiency score uses `(prompt - cached) + completion`. Change is `100 * (with - without) / without`; negative means fewer tokens. N/A comparisons include their reason in the coverage column; coverage is never estimated.

## Tier Status

| Tier | Purpose | Status | Evidence |
|---|---|---|---|
| Tier 1 | Static validation | **PASSED WITH OBSERVATIONS** | 11 validator(s); 8 finding(s) |
| Tier 2 | Semantic deduplication | **PASSED** | 2 validator(s); 0 finding(s) |
| Tier 3 | Live agent evaluation | **PASS** | 2 agent(s); 3 task(s) |

## Findings and Observations

<details>
<summary>Show detailed findings and successful checks</summary>

- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution: subprocess.run(
            [executable, *args], capture_output=True, text=True, timeout=15, check=False
        ) (`scripts/collect_preflight.py:100`)
- **MEDIUM** SECURITY/subprocess module call (AST4): Dangerous Code Execution: subprocess.run(command, capture_output=True, text=True, check=False) (`scripts/summarize_nsys.py:101`)
- **LOW** QUALITY/quality_reliability: Inputs are used but no dedicated Inputs section is documented (`skills/data-loading-bottleneck/SKILL.md`)
- **LOW** QUALITY/quality_reliability: Structured output is used but no dedicated Output Format section is documented (`skills/data-loading-bottleneck/SKILL.md`)
- **LOW** SCRIPT_LINT/magic_numbers: collect_preflight.py contains magic numbers (`skills/data-loading-bottleneck/scripts/collect_preflight.py`)
- 3 additional finding(s) are available in the full evaluation artifacts.

</details>

## Scoring Methodology

<details>
<summary>Show dimension definitions, source signals, and thresholds</summary>

| Dimension | Question | Scored signals |
|---|---|---|
| Security | Is it safe to use? | `security` (100%) |
| Correctness | Is the answer correct? | `accuracy` (100%) |
| Discoverability | Was the right skill loaded when needed? | `skill_execution` (100%) |
| Effectiveness | Did the skill help complete the task? | `goal_accuracy` (50%) + `behavior_check` (50%) |
| Efficiency | Did it avoid wasted tool calls and token usage? | `skill_efficiency` (50%) + `token_efficiency` (50%) |

- Dimension bands: PASS at 50% or above; NEUTRAL from 40% to below 50%; FAIL below 40%.
- Overall Tier 3 lift: PASS at +5 points or more; FAIL at -10 points or less; values between those bands are NEUTRAL.
- Overall verdict: PASS only when every configured dimension passes for at least one supported agent. Lift is reported as diagnostic evidence and does not override this gate.
- The 50% attempt pass threshold is a separate per-task gate; it is not the dimension pass threshold.
- Effectiveness is the equal-weight mean of goal completion (`goal_accuracy`) and expected workflow adherence (`behavior_check`).
- Efficiency is 50% tool-call productivity (the backward-compatible `skill_efficiency` wire id) and 50% `token_efficiency`. Positive-case skill routing is scored under Discoverability, not Efficiency; a negative case without a routing target is N/A. N/A sources are omitted, remaining weights are renormalized, and the dimension is marked partial.

Signals present in this run:

- `security` (Security): unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): whether the expected skill was selected, decoys were avoided, and the workflow executed.
- `skill_efficiency` (Tool Productivity): tool-call productivity (legacy wire id; routing is scored under Discoverability).
- `accuracy` (Accuracy): final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): whether the user's goal was achieved.
- `behavior_check` (Behavior Check): whether the expected workflow behavior was followed.
- `token_efficiency` (Token Efficiency): actual uncached prompt plus completion usage (50% of Efficiency).

</details>

## Freshness

Regenerate this benchmark when the skill, evaluation dataset, target agent/model, evaluator version, environment, or scoring policy changes.
