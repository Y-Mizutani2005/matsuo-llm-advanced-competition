# Matsuo Lab LLM Advanced Competition Snapshot

Portfolio snapshot of my U29 Excellence Award work in the Matsuo Lab LLM Course Advanced Competition.
The project improved an agent benchmark setup across DBBench and ALFWorld by fine-tuning Qwen2.5-7B-Instruct with SFT, LoRA / QLoRA, and data-format normalization.
The central idea was to convert ALFWorld function-calling trajectories into text ReAct format before hybrid SFT, so one model could learn both task families more consistently.
The best run improved the combined score from the no-SFT baseline while avoiding raw dataset, model artifact, credential, and large-log publication.

## Result Summary

Best run: `hybrid_alf_react`.

| Experiment | ALFWorld | DBBench | Combined | Submission Score |
|---|---:|---:|---:|---:|
| Baseline, no SFT | 48.0% | 51.0% | 99.0 | - |
| DB-only low-impact SFT | 56.0% | 52.405% | 108.405 | 4.1694 |
| Hybrid ALF ReAct SFT | 64.0% | 48.806% | 112.806 | 4.3387 |
| Hybrid ALF ReAct + MAX x2 | 64.0% | 48.715% | 112.715 | 4.3352 |

The final direction prioritized total agent performance over DB-only score. The hybrid run improved ALFWorld by 16.0 points versus baseline, while the DBBench regression stayed small enough for the combined score to improve.

## What I Built

- A profile-switchable SFT training script for DB-only and hybrid runs.
- DBBench role normalization from `agent` to `assistant`.
- Filtering for known malformed SQL samples with trailing `))`.
- ALFWorld function-calling to text ReAct conversion.
- Runbooks and experiment logs for the final training choices.
- Small analysis scripts for format evidence and DBBench error review.

## How To Read This Repo

1. [`experiment_log.md`](experiment_log.md): score progression and experiment decisions.
2. [`docs/technical-notes.md`](docs/technical-notes.md): short explanation of the core approach.
3. [`final_three_experiments_runbook.md`](final_three_experiments_runbook.md): final experiment profiles.
4. [`standard_code_sft_v2.py`](standard_code_sft_v2.py): main training script.
5. [`merge_and_upload.py`](merge_and_upload.py): sanitized model merge and upload helper.

## Repository Map

- `standard_code_sft_v2.py`: main Colab-oriented SFT script with experiment profile switching.
- `merge_and_upload.py`: environment-variable based helper for merging a LoRA adapter and uploading a merged model.
- `extract_format_evidence.py`: local log inspection helper for ALFWorld format comparison.
- `analyze_v4_errors.py`: local log inspection helper for DBBench task-limit errors.
- `experiment_log.md`: experiment history and score progression.
- `score_improvement_plan.md`: initial improvement strategy.
- `final_three_experiments_runbook.md`: final experiment profiles and execution order.
- `docs/development_document_simple_draft.md`: Japanese short development note.
- `docs/technical-notes.md`: polished technical summary for reviewers.

## Award And Evidence

- Award: Advanced Competition U29 Excellence Award, Matsuo Lab LLM Course.
- Public course context: [Matsuo-Iwasawa Lab LLM course page](https://weblab.t.u-tokyo.ac.jp/lecture/course-list/large-language-model/).
- Official result roster: [Matsuo Lab lecture Notion page, 大規模言語モデル2025 section](https://matsuolab-lecture.notion.site/432e287af97445d5aba989553ebaf808#126cfa7cece780d0807cea09481a604c).
- Certificate PDF: [`docs/evidence/matsuo-llm-advanced-u29-award-certificate.pdf`](docs/evidence/matsuo-llm-advanced-u29-award-certificate.pdf).

<img src="docs/evidence/matsuo-llm-advanced-u29-award-certificate.png" alt="U29 Excellence Award certificate for Matsuo Lab LLM Advanced Competition" width="560">

## What Is Intentionally Excluded

This snapshot does not include:

- API tokens, Hugging Face tokens, or local `.env` files.
- model checkpoints, LoRA adapter exports, or merged model weights.
- private submission credentials.
- raw datasets or redistribution-unclear data.
- generated caches, notebook outputs, and large logs.

## Reproducibility Note

The scripts are preserved to show engineering intent and experiment structure. They are not a one-command reproduction package because the original competition environment, model artifacts, local logs, and some data access assumptions are intentionally excluded from the public snapshot.
