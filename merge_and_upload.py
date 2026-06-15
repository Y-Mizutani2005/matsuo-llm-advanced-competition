"""
Merge a LoRA adapter into a base model and upload the merged model to Hugging Face.

This public snapshot version is configured through environment variables and does
not contain personal repository names, local paths, or tokens.
"""

import os
import shutil
from pathlib import Path

import torch
from huggingface_hub import HfApi, login, upload_folder
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


BASE_MODEL_ID = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen2.5-7B-Instruct")
ADAPTER_REPO_ID = os.environ.get("ADAPTER_REPO_ID")
MERGED_REPO_ID = os.environ.get("MERGED_REPO_ID")
TMP_MERGED_DIR = Path(os.environ.get("TMP_MERGED_DIR", "/tmp/merged_model"))
HF_TOKEN = os.environ.get("HF_TOKEN")
HF_PRIVATE_REPO = os.environ.get("HF_PRIVATE_REPO", "False").lower() in {"1", "true", "yes"}


def require_env(name: str, value: str | None) -> str:
    if not value:
        raise RuntimeError(f"{name} environment variable is required.")
    return value


def main() -> None:
    adapter_repo_id = require_env("ADAPTER_REPO_ID", ADAPTER_REPO_ID)
    merged_repo_id = require_env("MERGED_REPO_ID", MERGED_REPO_ID)
    hf_token = require_env("HF_TOKEN", HF_TOKEN)

    login(token=hf_token, add_to_git_credential=False)

    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_ID,
        trust_remote_code=True,
        token=hf_token,
    )
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
        token=hf_token,
    )

    model = PeftModel.from_pretrained(
        model,
        adapter_repo_id,
        token=hf_token,
    )
    model = model.merge_and_unload()

    TMP_MERGED_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(TMP_MERGED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(TMP_MERGED_DIR)

    api = HfApi()
    api.create_repo(
        repo_id=merged_repo_id,
        repo_type="model",
        exist_ok=True,
        private=HF_PRIVATE_REPO,
        token=hf_token,
    )

    upload_folder(
        folder_path=str(TMP_MERGED_DIR),
        repo_id=merged_repo_id,
        repo_type="model",
        commit_message=f"Upload merged model: {BASE_MODEL_ID} + LoRA adapter",
        token=hf_token,
    )

    shutil.rmtree(TMP_MERGED_DIR, ignore_errors=True)
    print(f"Uploaded merged model to https://huggingface.co/{merged_repo_id}")


if __name__ == "__main__":
    main()
