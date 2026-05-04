# DALI Review Philosophy

Distilled from ~8,500 historical review comments across the DALI maintainers. These notes complement the structured rules in `config.json` and shape the *style* of the review.

## Don't just scan the diff — understand the context

A change that looks correct in isolation can break invariants elsewhere.

- **Read complete files**, not just changed lines. Hidden assumptions live in the unchanged neighborhood.
- **Trace callers and callees** of any modified function. Renames, signature changes, and contract changes ripple outward.
- **Read the corresponding `.cc`/`.cu` for a changed `.h`** (and vice versa). Implementation detail often invalidates header-only assumptions.
- **Check whether a flagged pattern was introduced by this PR or already existed.** If pre-existing, say so explicitly — don't call it a new bug.

## Categorize findings

Prefix each comment so the author can prioritize:

- **[Critical]** — correctness bug, memory safety, race condition, or data loss.
- **[Bug]** — likely defect under realistic inputs (off-by-one, sign error, wrong enum, copy-paste mistake).
- **[Perf]** — measurable slowdown or unnecessary allocation/sync in a hot path.
- **[Style]** — convention or readability issue covered by an existing rule.
- **[Nit]** — minor preference, no real impact.
- **[Question]** — something you don't understand, want the author to confirm.

## Architecture concerns are judgment calls

If you find a high-level design issue (wrong abstraction, feature in wrong layer, design that creates maintenance pain), surface it as a **[Question]** or a top-level review comment — don't unilaterally demand a redesign in an inline note. These benefit from human discussion.

## Test review specifics

A test that "passes when the operator returns all zeros" is not a test. Look for:
- Comparison against a reference implementation (NumPy, OpenCV, torchvision).
- Coverage of `batch_size=1`, empty input, boundary sizes, both CPU and GPU variants.
- Deterministic seeds (no flaky randomness).
- `assert_raises(..., glob=...)` with a message pattern, not just exception type.
- No silent skips when a feature is unavailable in CI.

## Performance review specifics

DALI's hot paths get exercised at thousands of frames per second. Watch for:
- Unnecessary `cudaStreamSynchronize` calls — every one needs justification.
- Default-stream launches (`<<<g, b>>>(...)` without an explicit stream) — they create global sync points.
- Per-sample work that could be hoisted to constructor or `Setup` (file opens, decoder construction, attribute lookups).
- `std::string` from string literals where `string_view` or `const char*` would do.
- Container reallocation in inner loops (`reserve` when the size is known).
- Falling back to single-sample processing inside a batched code path.

## Error message standards

Every error must:
1. Name what the user did wrong (the parameter, the operator, the offending value).
2. Show expected vs. actual when relevant.
3. Tell the user how to fix it when the valid set is small.

Use `make_string(...)` for C++ concatenation, f-strings for Python. End sentences with a period.

## Be charitable, be concrete

- Don't repeat findings already raised by other reviewers (human or bot).
- Don't post nits when there are unresolved correctness concerns above — they drown out the signal.
- When suggesting a change, show the suggested code (or pseudocode) so the author can act on it.

## Review-of-review discipline

Before posting a comment, ask:
1. Is this a real issue, or a stylistic preference disguised as one?
2. Did this PR introduce it, or has it been there for years?
3. Does the comment contain a *suggestion*, or only a complaint?

If the answers are "preference / pre-existing / complaint only", skip it.
