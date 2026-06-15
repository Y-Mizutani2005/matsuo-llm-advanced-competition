# Final Three Experiments Runbook

## Profiles

`db_low_impact`
- Purpose: safest DB-only run that tries to preserve ALFWorld.
- Data: DBBench v1-v4 only
- Hyperparameters: LR=`2e-6`, epoch=`1`, MAX upsampling=`off`

`db_max2`
- Purpose: local fix for aggregation-MAX without the strong forgetting seen before.
- Data: DBBench v1-v4 only
- Hyperparameters: LR=`2e-6`, epoch=`1`, MAX upsampling=`2x`

`hybrid_alf_react`
- Purpose: final gamble. Mix DB with ALF after converting ALF function-calling trajectories into text ReAct.
- Data: DBBench v1-v4 + ALFWorld v5
- Hyperparameters: LR=`1e-6`, epoch=`1`, ALF react conversion=`on`

## Run Commands

Run each training job by setting `SFT_EXPERIMENT_PROFILE` before executing the training notebook/script cells in Colab.
If you want automatic Hugging Face repo names, also set `HF_UPLOAD_USER` once.

```python
import os
os.environ["HF_UPLOAD_USER"] = "your-hf-username"
os.environ["SFT_EXPERIMENT_PROFILE"] = "db_low_impact"
```

```python
import os
os.environ["HF_UPLOAD_USER"] = "your-hf-username"
os.environ["SFT_EXPERIMENT_PROFILE"] = "db_max2"
```

```python
import os
os.environ["HF_UPLOAD_USER"] = "your-hf-username"
os.environ["SFT_EXPERIMENT_PROFILE"] = "hybrid_alf_react"
```

## Expected Output Dirs

- `db_low_impact`: `/content/drive/MyDrive/lora_agentbench_qwen25_7b_db_low_impact`
- `db_max2`: `/content/drive/MyDrive/lora_agentbench_qwen25_7b_db_max2`
- `hybrid_alf_react`: `/content/drive/MyDrive/lora_agentbench_qwen25_7b_hybrid_alf_react`

Merged model dirs:

- `db_low_impact`: `/content/merged_model_db_low_impact`
- `db_max2`: `/content/merged_model_db_max2`
- `hybrid_alf_react`: `/content/merged_model_hybrid_alf_react`

Auto-generated HF repo ids when `HF_UPLOAD_USER=your-hf-username`:

- `db_low_impact`: `your-hf-username/agentbench-qwen25-7b-db-low-impact`
- `db_max2`: `your-hf-username/agentbench-qwen25-7b-db-max2`
- `hybrid_alf_react`: `your-hf-username/agentbench-qwen25-7b-hybrid-alf-react`

## Execution Order

1. Start `db_low_impact` and `db_max2` first.
2. Start `hybrid_alf_react` as the last gamble once the first two are safely queued.
3. At night, run inference for all three in parallel.
4. Prefer the model with the best combined DBBench + ALFWorld total score, not the best DB-only score.

## What Changed In Code

- Added profile switching via `SFT_EXPERIMENT_PROFILE`
- Added automatic removal of DBBench rows containing the known SQL `))` bug
- Added ALFWorld function-calling to text ReAct conversion for the hybrid gamble run
- Wired `neftune_noise_alpha` into `TrainingArguments` so the setting is no longer silently ignored
