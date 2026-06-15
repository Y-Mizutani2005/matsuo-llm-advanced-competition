# Runbook

This repo is a public snapshot. It documents the final experiment structure, but does not include raw datasets, model artifacts, local logs, or credentials.

## Final Profiles

Set `SFT_EXPERIMENT_PROFILE` before running the training script in Colab:

```python
import os
os.environ["SFT_EXPERIMENT_PROFILE"] = "db_low_impact"
```

```python
import os
os.environ["SFT_EXPERIMENT_PROFILE"] = "db_max2"
```

```python
import os
os.environ["SFT_EXPERIMENT_PROFILE"] = "hybrid_alf_react"
```

Optional upload settings:

```python
import os
os.environ["HF_UPLOAD_USER"] = "your-hf-username"
os.environ["HF_PRIVATE_REPO"] = "False"
```

## Expected Review Order

1. Review `experiment_log.md` to understand the score progression.
2. Review `docs/technical-notes.md` for the core format-normalization idea.
3. Review `final_three_experiments_runbook.md` for final profile settings.
4. Review `standard_code_sft_v2.py` for implementation details.

## Publication Checks

Before pushing changes, run:

```bash
git status --short
rg -n -i "(token|secret|password|api[_-]?key|hf_|openai|anthropic|azure|credential|Bearer|sk-|github_pat|ghp_)" .
```

Expected matches should be references to environment variable names or documentation warnings, not actual secret values.
