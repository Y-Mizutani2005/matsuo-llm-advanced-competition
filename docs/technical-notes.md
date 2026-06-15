# Technical Notes

## Problem

The Advanced Competition evaluated agent-like behavior across DBBench and ALFWorld. A simple SFT approach was not enough because the two task families used different conversation structures and action formats.

The main failure mode was negative transfer. A naive mix of DBBench and ALFWorld data reduced both scores because the model had to learn incompatible tool-use formats at the same time.

## Approach

The final approach used Qwen2.5-7B-Instruct with SFT and LoRA / QLoRA. The key engineering decision was to normalize ALFWorld examples before hybrid training.

For ALFWorld, assistant messages that used function-calling style actions were converted into text ReAct format:

```text
THOUGHT: inspect the environment and find the target object
ACTION: go to countertop 1
```

Tool outputs were then treated as the next observation. This made ALFWorld closer to the DBBench interaction pattern and reduced the format mismatch that hurt earlier hybrid runs.

## Iteration

The work was driven by error analysis:

- DBBench messages using `agent` role were normalized to `assistant`.
- Known malformed DBBench samples with trailing `))` were filtered.
- DB-only training was tested as a low-impact baseline.
- Aggregation-MAX upsampling was tested and rejected after it did not improve the best hybrid score.
- ALFWorld invalid actions were tracked as a major quality signal.

## Result

The best profile was `hybrid_alf_react`, which improved the combined score from `99.0` to `112.806`. It raised ALFWorld from `48.0%` to `64.0%`, while accepting a small DBBench drop from `51.0%` to `48.806%`.

The result was a better overall agent score, not a DB-only optimization.
