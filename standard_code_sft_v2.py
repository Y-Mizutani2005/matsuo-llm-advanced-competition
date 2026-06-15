# %% [markdown]
# # エージェント Trajectory SFT（Unsloth / Colab）標準学習コード：実行ガイド
# 
# 本ノートブックは、**AgentBench系タスク（ALFWorld・DBBench）のエージェント性能向上**を目的として、  
# Qwen2.5-7B-Instruct に対して **SFT（Supervised Fine-Tuning）** を行う標準コードです。
# 
# 学習は **Unsloth + LoRA / QLoRA** を利用し、Colab GPU 環境で動作するように構成されています。

# %% [markdown]
# 
# ## 1. このコードが行うこと（概要）
# 

# %% [markdown]
# 
# このコードは大きく3段階で構成されています。
# 
# 1. **環境固定（依存パッケージのバージョン固定）**  
#    Colabの環境変化による不具合を避けるため、numpy/transformers/trl/unsloth等を特定バージョンで揃えます。
# 
# 2. **SFT（教師あり微調整）の実行**  
#    Hugging Face Hub 上のトラジェクトリデータセット（ALFWorld / DBBench）を読み込み、ベースモデルに LoRA アダプタを差し込み、学習します。  
#    学習の損失（loss）は **全assistant turn**にかかる設計です（マルチターンのtool-useパターンを学習させやすい）。
# 
# 3. **（任意）マージ済みモデルのHugging Faceへのアップロード**  
#    学習で得られた LoRA アダプタをベースモデルにマージし、HF Hub に保存できます。
# 
# ---

# %% [markdown]
# 
# 
# ## 2. 実行手順（最短手順）
# 

# %% [markdown]
# 
# ### Step 0: Colab の準備
# - ランタイムの種類を **GPU** に変更し、GPU が **L4** になっていることを確認してください。
# - 過去の実行で環境が壊れている場合は **Runtime > Factory reset** を推奨します。
# 
# ### Step 1: 依存関係インストール
# - `uv pip install` を上から順に実行します。
# - 実行後、バージョン表示が想定通りであることを確認します（`unsloth import OK` が出ること）。
# 
# ### Step 2: Hugging Face へログイン
# - Colabの秘密鍵サービス（🔑アイコン）にHFトークンを設定しておくと、自動でログインされます。
# - 秘密鍵が未設定の場合は警告が表示されます。
# 
# ### Step 3: 学習の実行
# - `main()` が呼ばれ、学習が開始します。
# - 学習中に `[LabelStats:train]` が表示されます。これは「loss対象トークンが極端にゼロになっていないか」の健康診断です。
# - 学習前に `filter_has_supervision` が実行され、教師信号のないサンプルが自動除外されます。
# 
# ### Step 4: 学習成果物の確認
# - 学習後、`OUT_LORA_DIR` に以下が保存されます（最低限）：
#   - `adapter_config.json`
#   - `adapter_model.safetensors`（または `adapter_model.bin`）
#   - tokenizer 関連ファイル
# 
# ### Step 5:（任意）モデルマージとアップロード
# - LoRAアダプタをベースモデルにマージして、完全な推論用モデルを生成できます。
# - マージ済みモデルをHugging Faceにアップロードできます。
# 
# ---

# %% [markdown]
# ## 3. 出力（何が生成されるか）
# 
# 

# %% [markdown]
# - `OUT_LORA_DIR`（例：`/content/drive/MyDrive/lora_agentbench_qwen25_7b_hybrid_alf_react`）に、
#   **LoRAアダプタ（差分重み）**が保存されます。
# - このアダプタをベースモデルに適用（またはマージ）して推論することで、ALFWorld・DBBenchタスクのスコア改善を狙います。
# 
# ---

# %% [markdown]
# 
# 
# ## 4. 学習データセットの説明
# 

# %% [markdown]
# 
# ### 4.1 データセット概要
# 本コードでは、以下の2種類のトラジェクトリデータセットを使用できます。
# 環境変数 `SFT_DATASET_ID` で切り替えます。
# 
# #### A. ALFWorld トラジェクトリデータセット
# - HF Dataset: `u-10bei/sft_alfworld_trajectory_dataset_v5`
# 
# ALFWorld（テキストベースの家庭内タスク環境）におけるエージェントのトラジェクトリです。
# エージェントが「フライパンをダイニングテーブルに置く」「ライトで時計を調べる」などの
# 家庭内タスクを、テキストコマンドの発行と環境からの観察の繰り返しで解決します。
# 
# - 行数：**約 2,502 件**（成功＋失敗トラジェクトリ）
# - タスクタイプ：Pick & Place / Clean & Place / Heat & Place / Cool & Place / Examine in Light / Pick Two & Place
# 
# #### B. DBBench トラジェクトリデータセット
# - HF Dataset: `u-10bei/sft_sql_dataset_trajectories_v5`
# 
# DBBench（AgentBenchのデータベース操作タスク）におけるエージェントのトラジェクトリです。
# ユーザーの質問に対して、スキーマ探索→SQL構築→実行→エラー回復→最終回答という
# ReActスタイルの対話フローを辿ります。
# 
# - 行数：**約 3,000 件**
# - DBスキーマ：6種（university_academic / online_retail / hospital_management / company_hr / music_streaming / city_library）
# - タスクタイプ：Querying / Analysis / Updating（3難易度レベル）
# - 対話形式：ReAct形式（50%）＋ function-calling形式（50%）
# - スキーマ探索付き：約60%、エラー回復付き：約15%
# 
# ### 4.2 共通のデータ形式
# 両データセットとも **マルチターンのmessages形式** です。
# `system`（システムプロンプト）→ `user`（質問・環境観察）→ `assistant`（思考＋アクション）→ `tool`（実行結果）の繰り返しで構成されます。
# 
# #### DBBench（ReAct形式）の例：
# ```json
# [
#   {"role": "system", "content": "You are a database agent that interacts with a MySQL database..."},
#   {"role": "user", "content": "Find all classrooms in 'Science Hall' with capacity > 100."},
#   {"role": "assistant", "content": "Thought: I should first explore the schema...\nAction: execute_sql\n```sql\nSHOW TABLES;\n```"},
#   {"role": "tool", "content": "Classrooms | Courses | Departments | ..."},
#   {"role": "assistant", "content": "Thought: Now let me query the Classrooms table...\nAction: execute_sql\n```sql\nSELECT room_number, capacity FROM Classrooms WHERE building_name = 'Science Hall' AND capacity > 100;\n```"},
#   {"role": "tool", "content": "room_number | capacity\n301 | 150\n..."},
#   {"role": "assistant", "content": "Thought: I have the results.\nAnswer: The classrooms in Science Hall with capacity > 100 are: Room 301 (150 seats)..."}
# ]
# ```
# 
# #### ALFWorld の例：
# ```json
# [
#   {"role": "user", "content": "You are in the middle of a room...Your task is to: put a hot apple in fridge."},
#   {"role": "assistant", "content": "Think: I need to find an apple first.\nAction: go to countertop 1"},
#   {"role": "tool", "content": "On the countertop 1, you see an apple 1, a knife 2."},
#   ...
# ]
# ```
# 
# ### 4.3 メインコンペの構造化出力データセットとの主な違い
# - **ターン数**：2ターン → 数ターン〜数十ターン（マルチターンtool-use）
# - **role**：user/assistant のみ → system/user/assistant/tool の4種
# - **学習対象**：最終応答のみ → 全assistant turn（中間ステップ含む）
# - **シーケンス長**：512 → 2048（長いトラジェクトリに対応）

# %% [markdown]
# # 実行コード

# %% [markdown]
# ## Step -1: Google Driveのマウント（最初に実行必須）

# %%
# ============================================================
# -1) Google Driveのマウント（最初に実行必須）
# ============================================================
# ランタイムが切断されても学習データを保持するため、
# Google Driveにデータを保存します。
#
# ⚠️ 重要: このセルは学習開始前に必ず実行してください。
# ⚠️ ランタイム再接続後も、アップロード前に再実行してください。

from google.colab import drive
drive.mount('/content/drive')

print("✅ Google Driveがマウントされました")
print("   保存先: /content/drive/MyDrive/")

# %% [markdown]
# ## Step 1:依存関係インストール

# %%
# ============================================================
# 0) 依存関係の固定（Colabの“環境ブレ”対策）
# ============================================================
# Colab（無料版）は、ある日突然プリインストール版が変わり、
# それまで動いていた学習コードが壊れることが頻繁にあります。
# そのため、このセルでは「一度全部消す → 互換が確認できたバージョンを入れ直す」
# という“強制的な再現性確保”をしています。
#
# やりがちな事故：
# - 既に入っているパッケージと混ざって「importは通るが実行で落ちる」
# - transformers / trl / unsloth の相性不一致で、謎エラーや速度低下が起きる
#
# ※このセルは“おまじない”ではなく「再現性のための重要工程」です。

# Colab setup command:
# !uv pip install "numpy==2.0.2" "pandas==2.2.2"

# unsloth-zoo が要求する範囲に合わせる
# ここでは Unsloth と相性の良い transformers / trl / accelerate / peft / bitsandbytes を固定します。
# 特に transformers は細かいバージョン差で挙動が変わりやすいので固定が重要です。
# Colab setup command:
# !uv pip install \
#   "datasets==4.3.0" \
#   "trl==0.24.0" \
#   "transformers==4.56.2" \
#   "accelerate==1.4.0" \
#   "peft==0.13.2" \
#   "bitsandbytes==0.45.0"

# unsloth / zoo を同系列で揃える（zoo側の要求に合わせる）
# Unsloth本体と unsloth-zoo は“セット運用”が基本です。片方だけ上げると壊れがちです。
# Colab setup command:
# !uv pip install "unsloth-zoo==2025.12.7" "unsloth==2025.12.7"



# %%
# ============================================================
# 0.1) バージョン確認（“動くはず”の状態かを目視で確認）
# ============================================================
# ここで想定バージョンとズレている場合、
# 後工程で原因不明のエラーが出る確率が一気に上がります。

from unsloth import FastLanguageModel
import numpy as np, pandas as pd
import datasets, trl, transformers, torch

print("numpy", np.__version__)
print("pandas", pd.__version__)
print("datasets", datasets.__version__)
print("trl", trl.__version__)
print("transformers", transformers.__version__)
print("torch", torch.__version__)

print("unsloth import OK")

# 期待値：
# numpy 2.0.2
# pandas 2.2.2
# datasets 4.3.0（または <4.4.0 で 4.0.* / 4.1.0 以外）
# trl 0.24.0（または 0.18.2〜0.24.0 で 0.19.0以外）
# unsloth import OK


# -----------------------------
# 0) Install (single cell)
# -----------------------------
# NOTE:
# - Colabは初期状態が頻繁に変わるため、ピン留めで安定化します。
#   もし依存関係が壊れている環境であれば、Runtime > Factory reset を推奨。

# このセルを実行して、上の「期待値」にもしなっていない場合は、下記のコメントアウトを外して実行してみてください。
# !uv pip install \
#   "numpy==2.0.2" \
#   "pandas==2.2.2" \
#   "datasets==4.3.0" \
#   "trl==0.24.0" \
#   "transformers==4.57.3" \又は、4.56.2
#   "accelerate==1.4.0" \
#   "peft==0.13.2" \
#   "bitsandbytes==0.45.0" \
#   "unsloth-zoo==2025.12.7" \
#   "unsloth==2025.12.7" \
#   "huggingface_hub"


# %% [markdown]
# ## Step 2: HuggingFace ログイン
# 
# Colabの秘密鍵サービス（左パネルの🔑アイコン）に `HF_TOKEN` を設定しておくと、自動的にログインされます。
# 秘密鍵サービスの設定手順：
# 1. Hugging Face にログイン（https://huggingface.co/）
# 2. Settings → Access Tokens からトークンを作成（Write権限推奨）
# 3. Colabの左パネル 🔑 → 新しい秘密鍵を追加 → 名前: `HF_TOKEN`、値: コピーしたトークン

# %%
import os
from google.colab import userdata

# HF_TOKENを環境変数またはColabの秘密鍵から取得
# Colabの左側のパネルにある「秘密鍵（🔑）」アイコンをクリックし、
# 「HF_TOKEN」という名前でトークンを設定してください。
HF_TOKEN = os.environ.get('HF_TOKEN')
if not HF_TOKEN:
    HF_TOKEN = userdata.get('HF_TOKEN') # 秘密鍵サービスに登録した名前に変更

if not HF_TOKEN:
    print("警告: HF_TOKENが見つかりません。Hugging Faceへのログインが失敗する可能性があります。")

from huggingface_hub import login
login(token=HF_TOKEN, add_to_git_credential=True)

# %%

# -----------------------------
# 1) ライブラリ再読み込み
# -----------------------------
# Cell 12 で既にimport済みですが、Colabでセルを個別に実行する場合に備えて
# 必要なライブラリを再度importしています。

from unsloth import FastLanguageModel
import numpy as np, pandas as pd
import datasets, trl, transformers, torch

# %% [markdown]
# ## Step3:学習の実行

# %% [markdown]
# ============================================================  
# 2) Training code  
# ============================================================  
# ここからがSFT本体です。
# 大まかな流れ：
# 1) 設定値（モデル名、データセット、LoRA設定、学習率など）を読み込む
# 2) データセットをHFから取得し、必要な形（messages形式）を満たすものだけ残す
# 3) tokenizerで「学習に使うテキスト」を作ってキャッシュする（高速化）
# 4) ベースモデルを4bitでロードし、LoRAアダプタを差し込む
# 5) Trainerで学習を回す
# 6) LoRAアダプタを保存する

# %% [markdown]
# - ベースモデル：Qwen3-4B-Instruct-2507
# - GPU：L4（24GB VRAM、bf16ネイティブ対応）を前提としています。
# - 学習方式：LoRA（ベースモデルをfull精度でロードし、LoRAアダプタのみ学習）
#   - `load_in_4bit=False` でベースモデルを読み込み、LoRAアダプタ（軽量差分）だけを学習します。
#   - L4はbf16をネイティブサポートしているため、`bf16=True` で高速かつ安定した学習が可能です。
#   - そのため、学習後に保存されるのも「アダプタ」中心になります。

# %% [markdown]
# 
# 
# ---
# 
# 
# 
# 

# %% [markdown]
# ### 使用可能なデータセット
# 決勝用として、9種類のデータセットを用意しました。
# 環境変数 `SFT_DATASET_ID` を変更することで切り替えられます。
#   - この標準コードではデフォルトで1（ALFWorld）を使用しています。
#   - さらなる性能向上のため、データセットに追加で前処理を行ってから学習を行っても差し支えありません。
# 
# 1. ALFWorld：家庭内タスク  
# [u-10bei/sft_alfworld_trajectory_dataset](https://huggingface.co/datasets/u-10bei/sft_alfworld_trajectory_dataset)  
# [u-10bei/sft_alfworld_trajectory_dataset_v2](https://huggingface.co/datasets/u-10bei/sft_alfworld_trajectory_dataset_v2)  
# [u-10bei/sft_alfworld_trajectory_dataset_v3](https://huggingface.co/datasets/u-10bei/sft_alfworld_trajectory_dataset_v3)  
# [u-10bei/sft_alfworld_trajectory_dataset_v4](https://huggingface.co/datasets/u-10bei/sft_alfworld_trajectory_dataset_v4)   
#  [u-10bei/sft_alfworld_trajectory_dataset_v5](https://huggingface.co/datasets/u-10bei/sft_alfworld_trajectory_dataset_v5)
# 2. DBBench：データベース操作  
# [u-10bei/dbbench_sft_dataset_react](https://huggingface.co/datasets/u-10bei/dbbench_sft_dataset_react)  
# [u-10bei/dbbench_sft_dataset_react_v2](https://huggingface.co/datasets/u-10bei/dbbench_sft_dataset_react_v2)  
# [u-10bei/dbbench_sft_dataset_react_v3](https://huggingface.co/datasets/u-10bei/dbbench_sft_dataset_react_v3)  
# [u-10bei/dbbench_sft_dataset_react_v4](https://huggingface.co/datasets/u-10bei/dbbench_sft_dataset_react_v4)  

# %%
import os
import random
import json
import re
import shutil
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from datasets import load_dataset, Dataset
from transformers import TrainingArguments, Trainer, TrainerCallback


# %%
# ============================================================
# 学習後の自動処理設定
# ============================================================
# 学習完了後にGPUを自動解放するか（コンピューティングユニット節約）
AUTO_RELEASE_GPU = True

# Google Driveに保存するか（セッション落ち対策）
SAVE_TO_DRIVE = True

print(f"GPU自動解放: {'有効' if AUTO_RELEASE_GPU else '無効'}")
print(f"Google Drive保存: {'有効' if SAVE_TO_DRIVE else '無効'}")

# HuggingFaceアップロード設定
AUTO_UPLOAD_HF = True  # 学習後にHFへ自動アップロードするか

print(f"HF自動アップロード: {'有効' if AUTO_UPLOAD_HF else '無効'}")

# %%
# -----------------------------
# 環境変数の設定
# -----------------------------
# 下記の値を書き換えることで、コード本体を編集せずに設定を変更できます。

# 1. モデル・データセット関連
os.environ["SFT_EXPERIMENT_PROFILE"] = os.environ.get("SFT_EXPERIMENT_PROFILE", "db_low_impact")
os.environ["SFT_BASE_MODEL"] = "Qwen/Qwen2.5-7B-Instruct"  # 7Bモデル（上位参加者の主流）

# 複数データセットをカンマ区切りで指定（マルチデータセット対応）
# DBBenchのみSFT戦略: ALFWorldはベースモデルの汎用能力に任せ、DBBenchに全集中
# 理由: ALFWorldデータ（function-calling形式）とDBBench（テキストReAct形式）の
#       フォーマット不一致によるタスク間干渉（negative transfer）を排除するため
os.environ["SFT_DATASET_IDS"] = ",".join([
    "u-10bei/dbbench_sft_dataset_react_v4",          # DBBench v4: 1200件, テーブル400種（最多利用604モデル）
    "u-10bei/dbbench_sft_dataset_react_v3",          # DBBench v3: 1200件, テーブル252種
    "u-10bei/dbbench_sft_dataset_react_v2",          # DBBench v2: 360件, テーブル166種
    "u-10bei/dbbench_sft_dataset_react",             # DBBench v1: 300件, テーブル214種（重複少）
])
# 旧互換: 単一データセット指定も可能
os.environ["SFT_DATASET_ID"] = ""  # SFT_DATASET_IDSが優先される

# 出力先（Drive保存対応）
# 実験ごとにサフィックスを変更して混ざらないようにする
_dir_suffix = "_dbonly_neftune"
if SAVE_TO_DRIVE:
    os.environ["SFT_OUT_LORA_DIR"] = f"/content/drive/MyDrive/lora_agentbench_qwen25_7b{_dir_suffix}"
else:
    os.environ["SFT_OUT_LORA_DIR"] = f"/content/lora_agentbench_qwen25_7b{_dir_suffix}"

# マージ済みモデルも同様に分離
os.environ["MERGED_MODEL_DIR"] = f"/content/merged_model{_dir_suffix}"

# 2. 学習の基本パラメータ
os.environ["SFT_SEED"] = "3407"
os.environ["SFT_VAL_RATIO"] = "0.05"
os.environ["SFT_MAX_SEQ_LEN"] = "4096"  # 推論時8192に対応するため拡大

# --- HuggingFace アップロード先 ---
# ユーザー名だけ指定すれば、profileごとに別repo名を自動生成します。
os.environ["HF_UPLOAD_USER"] = os.environ.get("HF_UPLOAD_USER", "your-username")
os.environ["HF_UPLOAD_REPO_ID"] = os.environ.get("HF_UPLOAD_REPO_ID", "your-username/your-repo-name")
os.environ["HF_PRIVATE_REPO"] = os.environ.get("HF_PRIVATE_REPO", "False")  # True で非公開リポジトリ

# 3. LoRA (アダプタ) 設定
os.environ["SFT_LORA_R"] = "64"
os.environ["SFT_LORA_ALPHA"] = "128"
os.environ["SFT_LORA_DROPOUT"] = "0"
os.environ["SFT_LORA_TARGET_MODULES"] = "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj"

# 4. ハイパーパラメータ
os.environ["SFT_EPOCHS"] = "2"
os.environ["SFT_PER_DEVICE_TRAIN_BS"] = "1"   # L4 (24GB): QLoRA + seq4096でBS=1
os.environ["SFT_PER_DEVICE_EVAL_BS"] = "1"
os.environ["SFT_GRAD_ACCUM"] = "8"            # 実効バッチサイズ=8を維持
os.environ["SFT_LR"] = "1e-5"                 # tomtom氏実績あり。上位者は2e-5〜1e-5を使用
os.environ["SFT_WARMUP_RATIO"] = "0.1"
os.environ["SFT_WEIGHT_DECAY"] = "0.05"

# 5. ステップ・保存設定
os.environ["SFT_MAX_STEPS"] = "-1" # -1でエポックベース。動作確認時は 10 などに。
os.environ["SFT_LOGGING_STEPS"] = "10"
os.environ["SFT_EVAL_STEPS"] = "30"
os.environ["SFT_SAVE_STEPS"] = "100"
os.environ["SFT_SAVE_TOTAL_LIMIT"] = "2"

# 6. 特殊学習設定 (CoTマスク・アップサンプリング)
os.environ["SFT_MASK_COT"] = "0" # "1" で有効, "0" で無効
os.environ["SFT_OUTPUT_MARKERS"] = "Output:,OUTPUT:,Final:,Answer:,Result:,Response:"
os.environ["SFT_OUTPUT_LEARN_MODE"] = "after_marker" # "after_marker" または "from_marker"
os.environ["SFT_USE_UPSAMPLING"] = "0" # "1" で有効, "0" で無効  # データ2 専用
os.environ["SFT_UPSAMPLE_RULES"] = "" # 例: '{"pack:math": 2.0}' # データ2 専用

# 7. aggregation-MAX アップサンプリング（ベースラインで0%だった問題への対策）
#    "1" にするとDBBenchデータ内のMAX系クエリを自動検出し3倍に増幅
os.environ["SFT_UPSAMPLE_AGGREGATION_MAX"] = "0"  # "1" で有効, "0" で無効
os.environ["SFT_AGGREGATION_MAX_MULTIPLIER"] = "3" # アップサンプリング倍率

# 8. NEFTune (Noisy Embeddings Fine-Tuning)
#    SFT時の過学習を防ぎ、ベースモデルの汎化能力（ALFWorld含む）を維持する
#    推奨値: 5〜15。0で無効。
os.environ["SFT_NEFTUNE_NOISE_ALPHA"] = "5"

print("環境変数の設定が完了しました。")

# %%
def _getenv(name: str, default: str):
    return os.environ.get(name, default)

def _getenv_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except Exception:
        return default

def _getenv_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except Exception:
        return default


def apply_experiment_profile() -> None:
    profile = _getenv("SFT_EXPERIMENT_PROFILE", "db_low_impact").strip() or "db_low_impact"
    print(f"[PROFILE] Using experiment profile: {profile}")
    hf_user = _getenv("HF_UPLOAD_USER", "your-username").strip()
    hf_private = _getenv("HF_PRIVATE_REPO", "False")

    common = {
        "SFT_BASE_MODEL": "Qwen/Qwen2.5-7B-Instruct",
        "SFT_SEED": "3407",
        "SFT_VAL_RATIO": "0.05",
        "SFT_MAX_SEQ_LEN": "4096",
        "SFT_LORA_R": "64",
        "SFT_LORA_ALPHA": "128",
        "SFT_LORA_DROPOUT": "0",
        "SFT_LORA_TARGET_MODULES": "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        "SFT_PER_DEVICE_TRAIN_BS": "1",
        "SFT_PER_DEVICE_EVAL_BS": "1",
        "SFT_GRAD_ACCUM": "8",
        "SFT_WARMUP_RATIO": "0.1",
        "SFT_WEIGHT_DECAY": "0.05",
        "SFT_MAX_STEPS": "-1",
        "SFT_LOGGING_STEPS": "10",
        "SFT_EVAL_STEPS": "30",
        "SFT_SAVE_STEPS": "100",
        "SFT_SAVE_TOTAL_LIMIT": "2",
        "SFT_MASK_COT": "0",
        "SFT_OUTPUT_MARKERS": "Output:,OUTPUT:,Final:,Answer:,Result:,Response:",
        "SFT_OUTPUT_LEARN_MODE": "after_marker",
        "SFT_USE_UPSAMPLING": "0",
        "SFT_UPSAMPLE_RULES": "",
        "SFT_UPSAMPLE_AGGREGATION_MAX": "0",
        "SFT_AGGREGATION_MAX_MULTIPLIER": "2",
        "SFT_NEFTUNE_NOISE_ALPHA": "0",
        "SFT_ALF_REACT_CONVERSION": "0",
        "SFT_DB_CLEAN_PAREN_BUG": "1",
    }

    db_dataset_ids = ",".join([
        "u-10bei/dbbench_sft_dataset_react_v4",
        "u-10bei/dbbench_sft_dataset_react_v3",
        "u-10bei/dbbench_sft_dataset_react_v2",
        "u-10bei/dbbench_sft_dataset_react",
    ])
    db_plus_alf_ids = ",".join([
        "u-10bei/dbbench_sft_dataset_react_v4",
        "u-10bei/dbbench_sft_dataset_react_v3",
        "u-10bei/dbbench_sft_dataset_react_v2",
        "u-10bei/dbbench_sft_dataset_react",
        "u-10bei/sft_alfworld_trajectory_dataset_v5",
    ])

    profiles = {
        "db_low_impact": {
            "SFT_DATASET_IDS": db_dataset_ids,
            "SFT_DATASET_ID": "",
            "SFT_EPOCHS": "1",
            "SFT_LR": "2e-6",
            "_DIR_SUFFIX": "_db_low_impact",
        },
        "db_max2": {
            "SFT_DATASET_IDS": db_dataset_ids,
            "SFT_DATASET_ID": "",
            "SFT_EPOCHS": "1",
            "SFT_LR": "2e-6",
            "SFT_UPSAMPLE_AGGREGATION_MAX": "1",
            "SFT_AGGREGATION_MAX_MULTIPLIER": "2",
            "_DIR_SUFFIX": "_db_max2",
        },
        "hybrid_alf_react": {
            "SFT_DATASET_IDS": db_plus_alf_ids,
            "SFT_DATASET_ID": "",
            "SFT_EPOCHS": "1",
            "SFT_LR": "1e-6",
            "SFT_ALF_REACT_CONVERSION": "1",
            "_DIR_SUFFIX": "_hybrid_alf_react",
        },
        "hybrid_alf_react_max2": {
            "SFT_DATASET_IDS": db_plus_alf_ids,
            "SFT_DATASET_ID": "",
            "SFT_EPOCHS": "1",
            "SFT_LR": "1e-6",
            "SFT_ALF_REACT_CONVERSION": "1",
            "SFT_UPSAMPLE_AGGREGATION_MAX": "1",
            "SFT_AGGREGATION_MAX_MULTIPLIER": "2",
            "_DIR_SUFFIX": "_hybrid_alf_react_max2",
        },
    }

    if profile not in profiles:
        raise ValueError(f"Unknown SFT_EXPERIMENT_PROFILE: {profile}")

    merged = dict(common)
    merged.update(profiles[profile])

    dir_suffix = merged.pop("_DIR_SUFFIX")
    repo_suffix = dir_suffix.lstrip("_").replace("_", "-")
    merged["SFT_OUT_LORA_DIR"] = (
        f"/content/drive/MyDrive/lora_agentbench_qwen25_7b{dir_suffix}"
        if SAVE_TO_DRIVE else
        f"/content/lora_agentbench_qwen25_7b{dir_suffix}"
    )
    merged["MERGED_MODEL_DIR"] = f"/content/merged_model{dir_suffix}"
    merged["HF_PRIVATE_REPO"] = hf_private
    if hf_user and hf_user != "your-username":
        merged["HF_UPLOAD_REPO_ID"] = f"{hf_user}/agentbench-qwen25-7b-{repo_suffix}"
    else:
        merged["HF_UPLOAD_REPO_ID"] = "your-username/your-repo-name"

    for key, value in merged.items():
        os.environ[key] = value


apply_experiment_profile()

# 学習の“出発点”となるベースモデル（7B）
BASE_MODEL_ID = _getenv("SFT_BASE_MODEL", "Qwen/Qwen2.5-7B-Instruct")

# 学習に使うSFTデータセット（HF Hub上に置かれている想定）
# マルチデータセット: カンマ区切りで複数指定可能
DATASET_IDS_STR = _getenv("SFT_DATASET_IDS", "")
DATASET_ID      = _getenv("SFT_DATASET_ID", "u-10bei/sft_alfworld_trajectory_dataset_v5")

# DATASET_IDSが設定されている場合はそちらを優先
if DATASET_IDS_STR.strip():
    DATASET_IDS = [s.strip() for s in DATASET_IDS_STR.split(",") if s.strip()]
else:
    DATASET_IDS = [DATASET_ID]

# 学習後に保存されるLoRAアダプタの出力先（ローカル）
OUT_LORA_DIR  = _getenv("SFT_OUT_LORA_DIR", "/content/lora_agentbench_qwen25_7b")

# マージ済みモデルの保存先
MERGED_MODEL_DIR = _getenv("MERGED_MODEL_DIR", "/content/merged_model")

SEED        = _getenv_int("SFT_SEED", 3407)
VAL_RATIO   = _getenv_float("SFT_VAL_RATIO", 0.05)

# 1サンプルあたり最大何トークンまで見るか（長いほど情報を見られるが、GPUメモリと時間が増える）
MAX_SEQ_LEN = _getenv_int("SFT_MAX_SEQ_LEN", 2048)

# LoRA Config（＝“どれくらいの表現力を持つ差分を学習するか”）
LORA_R       = _getenv_int("SFT_LORA_R", 64)
LORA_ALPHA   = _getenv_int("SFT_LORA_ALPHA", 128)
LORA_DROPOUT = _getenv_float("SFT_LORA_DROPOUT", 0)
LORA_TARGET_MODULES = (
    _getenv("SFT_LORA_TARGET_MODULES", "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj").split(",")
)

# Train hyperparams（学習の基本設定）
NUM_TRAIN_EPOCHS            = _getenv_int("SFT_EPOCHS", 1)
PER_DEVICE_TRAIN_BATCH_SIZE = _getenv_int("SFT_PER_DEVICE_TRAIN_BS", 2)
PER_DEVICE_EVAL_BATCH_SIZE  = _getenv_int("SFT_PER_DEVICE_EVAL_BS", 2)

# 勾配累積：GPUに一度に載せられるバッチが小さい時に、複数ステップ分を貯めて“大きいバッチ相当”にする
GRAD_ACCUM                  = _getenv_int("SFT_GRAD_ACCUM", 4)

LR                          = _getenv_float("SFT_LR", 1e-6)
WARMUP_RATIO                = _getenv_float("SFT_WARMUP_RATIO", 0.1)

# Debug / quick check
# MAX_STEPSを小さくすると“動作確認だけ”の短時間学習ができます（本番は -1 のまま）
MAX_STEPS        = _getenv_int("SFT_MAX_STEPS", -1)
LOGGING_STEPS    = _getenv_int("SFT_LOGGING_STEPS", 10)
EVAL_STEPS       = _getenv_int("SFT_EVAL_STEPS", 30)
SAVE_STEPS       = _getenv_int("SFT_SAVE_STEPS", 100)
SAVE_TOTAL_LIMIT = _getenv_int("SFT_SAVE_TOTAL_LIMIT", 2)
WEIGHT_DECAY     = _getenv_float("SFT_WEIGHT_DECAY", 0.05)

# Optional: upsampling rules
# 特定のサブカテゴリ（例：難しいタスク）を“多めに学習させる”ための仕組み。
# 標準ではOFFになっています。
UPSAMPLE_ENABLE      = _getenv("SFT_USE_UPSAMPLING", "0") in ("1","true","True")
UPSAMPLE_RULES_JSON = _getenv("SFT_UPSAMPLE_RULES", "")

# aggregation-MAX アップサンプリング設定
UPSAMPLE_AGG_MAX = _getenv("SFT_UPSAMPLE_AGGREGATION_MAX", "0") in ("1","true","True")
AGG_MAX_MULTIPLIER = _getenv_int("SFT_AGGREGATION_MAX_MULTIPLIER", 3)

# NEFTune 設定
NEFTUNE_NOISE_ALPHA = _getenv_float("SFT_NEFTUNE_NOISE_ALPHA", 0)
ALF_REACT_CONVERSION = _getenv("SFT_ALF_REACT_CONVERSION", "0") in ("1", "true", "True")
DB_CLEAN_PAREN_BUG = _getenv("SFT_DB_CLEAN_PAREN_BUG", "1") in ("1", "true", "True")


# -----------------------------
# 2.2) Seed & Utils
# -----------------------------
# 乱数（シャッフルやサンプリング）を固定して、再現性を担保します。
# seedが同じなら、原則として同じ分割・同じ抽出になりやすいです。

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

seed_everything(SEED)


def normalize_roles(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    DBBenchデータセットのrole名を標準化する。
    DBBenchではrole名が "agent" だが、標準のchat templateとcollatorは
    "assistant" を期待するため、変換が必要。

    Args:
        msgs: messages形式のリスト [{"role": ..., "content": ...}, ...]

    Returns:
        role名が正規化されたmessagesリスト
    """
    normalized = []
    for msg in msgs:
        new_msg = dict(msg)  # コピーして元データを変更しない
        if new_msg.get("role") == "agent":
            new_msg["role"] = "assistant"
        normalized.append(new_msg)
    return normalized


def _extract_action_from_tool_calls(tool_calls: Any) -> str:
    if not isinstance(tool_calls, list):
        return ""
    for tc in tool_calls:
        fn = tc.get("function", {}) if isinstance(tc, dict) else {}
        args = fn.get("arguments", "")
        if isinstance(args, dict):
            action = args.get("action", "")
            if action:
                return str(action).strip()
        if isinstance(args, str) and args.strip():
            try:
                payload = json.loads(args)
                action = payload.get("action", "")
                if action:
                    return str(action).strip()
            except Exception:
                pass
    return ""


def _normalize_alf_thought_text(text: str) -> str:
    text = str(text or "").strip()
    if not text:
        return ""
    text = re.sub(r"^\s*Think\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*Thought\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*THOUGHT\s*:\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAction\s*:\s*.*$", "", text, flags=re.IGNORECASE | re.DOTALL).strip()
    return text


def convert_alfworld_messages_to_react(msgs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted = []
    for msg in msgs:
        role = msg.get("role")
        content = str(msg.get("content", "") or "").strip()

        if role == "assistant":
            action = _extract_action_from_tool_calls(msg.get("tool_calls"))
            thought = _normalize_alf_thought_text(content)
            if thought and action:
                content = f"THOUGHT: {thought}\nACTION: {action}"
            elif action:
                content = f"ACTION: {action}"
            elif thought:
                content = f"THOUGHT: {thought}"
            converted.append({"role": "assistant", "content": content})
            continue

        if role == "tool":
            converted.append({"role": "user", "content": content})
            continue

        if role == "agent":
            converted.append({"role": "assistant", "content": content})
            continue

        if role in ("user", "system"):
            converted.append({"role": role, "content": content})
            continue

    return converted


def looks_like_alfworld_dataset(dataset_id: str, ds: Dataset) -> bool:
    if "alfworld" in str(dataset_id).lower():
        return True
    if "tools" in ds.column_names:
        return True
    if "messages" not in ds.column_names or len(ds) == 0:
        return False
    try:
        sample = ds[0]["messages"]
    except Exception:
        return False
    if not isinstance(sample, list):
        return False
    return any(isinstance(m, dict) and m.get("role") == "tool" for m in sample)


def has_db_paren_bug(msgs: List[Dict[str, Any]]) -> bool:
    for msg in msgs:
        if msg.get("role") not in ("assistant", "agent"):
            continue
        content = str(msg.get("content", "") or "")
        if "```sql" in content and "))" in content:
            return True
    return False


# データセットをmessages形式に変換する関数
def convert_to_messages_format(example):
    """
    データセット形式に応じて適切にmessages形式に変換する。
    DBBenchのrole名 "agent" → "assistant" への変換も行う。

    Args:
        example: データセットの1行（dict）

    Returns:
        dict: {"messages": [{"role": ..., "content": ...}, ...]} 形式
    """
    # 既にmessages形式ならそのまま返す（trajectory形式データセット）
    if "messages" in example and isinstance(example["messages"], list):
        msgs = example["messages"]
        # messages内の各要素が {role, content} を持つか確認
        if len(msgs) > 0 and isinstance(msgs[0], dict) and "role" in msgs[0]:
            # role名の正規化（agent → assistant）
            msgs = normalize_roles(msgs)
            if not hasattr(convert_to_messages_format, "_logged"):
                print("--- Debug: Dataset already in messages format (trajectory) ---")
                print(f"  Turns: {len(msgs)}")
                print(f"  Roles: {[m['role'] for m in msgs]}")
                convert_to_messages_format._logged = True
            return {"messages": msgs}

    # 旧形式: question/SQL or instruction/output から変換
    q = example.get('question') or example.get('instruction', "")
    inp = example.get('input', "")
    if inp and isinstance(inp, str) and inp.strip():
        q += f"\nSchema: {inp}"

    a = example.get('SQL') or example.get('output', "")

    if not hasattr(convert_to_messages_format, "_logged"):
        print("-- Debug: Converting from legacy format --")
        print(f"Q: {str(q)[:100]}...")
        print(f"A: {str(a)[:100]}...")
        convert_to_messages_format._logged = True

    user_content = f"You are a database expert. Your task is to generate a SQL query based on the provided question.\nQuestion: {q}"
    assistant_content = f"Output: {a}"

    return {
        "messages": [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": assistant_content}
        ]
    }

def ensure_openai_messages(ds: Dataset, msg_col: str = "messages") -> None:
    # データが「messages: [{role, content}, ...]」形式かをチェックします。
    # これは ChatGPT形式（OpenAIのChat Completions形式に似た）で、
    # tokenizer.apply_chat_template で安全に文字列化するために必要です。
    row0 = ds[0]
    ex = row0.get(msg_col, None)
    if not isinstance(ex, list):
        raise ValueError(f"Dataset must have list-style 'messages'. Got {type(ex)}")

def has_any_nonempty_assistant_turn(msgs: List[Dict[str, Any]]) -> bool:
    # “assistantの発話が空じゃない”ものが1回でも含まれるか？
    # SFTでは「正解例（assistantの出力）」がないと学習できないため。
    return any(m.get("role") == "assistant" and str(m.get("content", "")).strip() != "" for m in msgs)

def ends_with_nonempty_assistant(ex: Dict[str, Any]) -> bool:
    # トラジェクトリデータセットでは最後のメッセージがtoolであることがあるため、
    # 厳密に「最後のターンがassistant」である必要はないと判断し、
    # 学習可能なassistantターンが一つでもあればTrueを返すように変更します。
    # (has_any_nonempty_assistant_turnが既にこのチェックを行っています)
    # これにより、ValueError: num_samples=0 の問題を回避します。
    msgs = ex.get("messages", [])
    if not msgs:
        return False # No messages, not valid

    # 会話全体の中で空でないアシスタントターンがあるかどうかを確認します。
    # 少なくとも 1 つあり、照合者がすべてのアシスタントターンを処理する場合、
    # この例はトレーニングに使用できます。
    return has_any_nonempty_assistant_turn(msgs)

def shuffle_split(ds: Dataset, val_ratio: float, seed: int) -> Tuple[Dataset, Dataset]:
    # データをシャッフルして train/val に分割します。
    # val（検証）を持つことで「学習が進むほど性能が上がっているか／過学習していないか」を見られます。
    ds_shuf = ds.shuffle(seed=seed)
    n = len(ds_shuf)
    n_val = max(1, int(round(n * val_ratio)))
    return ds_shuf.select(range(n_val, n)), ds_shuf.select(range(n_val))

def make_text_cache_builder(tokenizer):
    # messages形式 → 実際にモデルに入力する“1本のテキスト”へ変換する関数を作ります。さらに「トークン長（truncationなし）」もキャッシュします。
    #
    # full_text  : ユーザー＋アシスタント（正解）まで含んだ全文
    # prefix_text: “最後のassistantの直前まで”の文（＝ここからassistantを生成させたい）
    #
    # この2つを持つことで、後のcollatorで「assistant部分だけをloss対象にする境界」を計算できます。

    def _build(batch):
        full_out = []
        prefix_out = []
        full_len_out = []
        prefix_len_out = []

        for msgs in batch["messages"]:
            full = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
            prefix = tokenizer.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)

            full_out.append(full)
            prefix_out.append(prefix)

            # 重要：ここで truncation=False で token 長だけ計算してキャッシュする
            # add_special_tokens=False はあなたの現行設計に合わせる（テンプレ側で必要トークンが入る想定）
            full_ids = tokenizer(full, add_special_tokens=False, truncation=False)["input_ids"]
            prefix_ids = tokenizer(prefix, add_special_tokens=False, truncation=False)["input_ids"]

            full_len_out.append(len(full_ids))
            prefix_len_out.append(len(prefix_ids))

        return {
            "full_text": full_out,
            "prefix_text": prefix_out,
            "full_input_ids_len": full_len_out,
            "prefix_input_ids_len": prefix_len_out,
        }

    return _build



# -----------------------------
# 2.3) Collator (assistant-only loss)
# -----------------------------
# collatorは「生のサンプル群 → 学習に必要なテンソル(input_ids/labels等)」に変換する部品です。
#
# ここがこの学習コードの“設計思想”の核心：
# - 入力（user/system）も含めてモデルには読ませる
# - ただし loss（誤差）を計算するのは assistant の出力部分だけ
#
# これにより：
# - 「プロンプトを丸暗記させる」方向に学習が引っ張られにくい
# - “回答の形式”や“出力の正確さ”に学習の力点を置ける
#
# ALFWorldやDBBenchのようなマルチターンtool-useタスクでは、この設計は合理的です。

# 使用データセットによる仕様の違い
# データセット1：Output: が 100% なので CoT マスクが常に動き、Output本体だけ学習
# データセット2：Output: 系ラベルが存在しないため、CoTマスクは発動せず、“出力本体”を学習

# --- CoT mask settings (env overridable) ---
MASK_COT = _getenv("SFT_MASK_COT", "1") in ("1","true","True")
OUTPUT_MARKERS = [s.strip() for s in _getenv(
    "SFT_OUTPUT_MARKERS",
    "Output:,OUTPUT:,Final:,Answer:,Result:,Response:"
).split(",") if s.strip()]
OUTPUT_LEARN_MODE = _getenv("SFT_OUTPUT_LEARN_MODE", "after_marker")  # after_marker / from_marker

import torch
from dataclasses import dataclass
from typing import Any, Dict, List

@dataclass
class AssistantOnlyCollatorCached:
    """
    全てのassistant turnにlossをかけるCollator。

    旧版との違い:
    - 旧: 最後のassistant turnのみloss → trajectory中間ステップが学習されない
    - 新: 全assistant turnにloss → 環境観察、アクション選択、SQL構築、リカバリーを全て学習

    動作原理:
    1. 各サンプルのmessagesを chat_template で1本のテキストに変換
    2. 各assistant turnの開始・終了位置を特定
    3. assistant部分のみlabelsを設定、それ以外は-100（loss計算外）
    """
    tokenizer: Any
    max_length: int = 2048

    def _apply_template(self, msgs, tools=None, add_generation_prompt=False):
        """apply_chat_template のラッパー。tools を統一的に渡す。"""
        kwargs = dict(
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
        )
        # tools がある場合のみ渡す（無い場合に渡すと template がエラーになる場合がある）
        if tools:
            kwargs["tools"] = tools
        return self.tokenizer.apply_chat_template(msgs, **kwargs)

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        tok = self.tokenizer

        all_input_ids = []
        all_labels = []
        all_attention = []

        for ex in batch:
            msgs = ex["messages"]
            tools = ex.get("tools", None)  # function-calling 形式のツール定義

            # 全メッセージを1本のテキストに変換（tools 付き）
            full_text = self._apply_template(
                msgs, tools=tools, add_generation_prompt=False
            )

            # truncation なしの全長を取得（境界計算用）
            full_ids_notrun = tok(
                full_text, add_special_tokens=False,
                truncation=False
            )["input_ids"]

            # truncation ありでエンコード（実際の入力用）
            full_ids = tok(
                full_text, add_special_tokens=False,
                truncation=True, max_length=self.max_length
            )["input_ids"]

            # left-truncationのオフセット
            trunc_offset = max(0, len(full_ids_notrun) - self.max_length)

            labels = [-100] * len(full_ids)

            # 各 assistant turn の境界を計算してlabelsを設定
            for i, msg in enumerate(msgs):
                if msg["role"] != "assistant":
                    continue

                # msgs[0..i-1] のテキスト長 = このassistant turnの開始位置
                prefix_text = self._apply_template(
                    msgs[:i], tools=tools, add_generation_prompt=True
                )
                prefix_ids = tok(
                    prefix_text, add_special_tokens=False, truncation=False
                )["input_ids"]

                # msgs[0..i] のテキスト長 = このassistant turnの終了位置
                through_text = self._apply_template(
                    msgs[:i+1], tools=tools, add_generation_prompt=False
                )
                through_ids = tok(
                    through_text, add_special_tokens=False, truncation=False
                )["input_ids"]

                # truncation補正後の位置
                start = max(0, len(prefix_ids) - trunc_offset)
                end = max(0, len(through_ids) - trunc_offset)

                # labelsにassistant部分のtoken IDをコピー
                for j in range(start, min(end, len(full_ids))):
                    if j >= 0:
                        labels[j] = full_ids[j]

            all_input_ids.append(full_ids)
            all_labels.append(labels)
            all_attention.append([1] * len(full_ids))

        # Padding (right padding)
        max_len = max(len(ids) for ids in all_input_ids)
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0

        padded_ids = []
        padded_labels = []
        padded_attention = []

        for i in range(len(all_input_ids)):
            pad_len = max_len - len(all_input_ids[i])
            padded_ids.append(all_input_ids[i] + [pad_id] * pad_len)
            padded_labels.append(all_labels[i] + [-100] * pad_len)
            padded_attention.append(all_attention[i] + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention, dtype=torch.long),
        }

# %%
import random, torch
from datasets import Dataset
from huggingface_hub import hf_hub_download
import json

@torch.no_grad()
def filter_has_supervision(ds, collator):
    keep = []
    for i in range(len(ds)):
        out = collator([ds[i]])
        if (out["labels"][0] != -100).sum().item() > 0:
            keep.append(i)
    return ds.select(keep)


def count_all_masked(ds, collator, n=200, seed=3407):
    rng = random.Random(seed)
    n = min(n, len(ds))
    idxs = [rng.randrange(0, len(ds)) for _ in range(n)]
    all_masked = 0
    for i in idxs:
        out = collator([ds[i]])
        labels = out["labels"][0]
        if (labels != -100).sum().item() == 0:
            all_masked += 1
    print(f"[CHECK] all-masked samples in {n}: {all_masked} ({all_masked/max(1,n):.1%})")

# -----------------------------
# 2.4) Callback (monitor)
# -----------------------------
# 学習中のデバッグ用コールバックです。
# ここでは「labelsのうち、実際にloss対象になっているトークン割合」を時々表示します。
#
# 意味：
# - valid_ratio が極端に小さい → “学習していない”のと同じ（ラベルがほぼ -100）
# - valid_ratio が適度にある → assistant部分にしっかりlossが乗っている
#
# 初学者向けに言うと：
# - これは“学習がちゃんと効いているかの健康診断”です。

class LabelStatsCallback(TrainerCallback):
    def __init__(self, dataset, collator, name="train", every_n_steps=100):
        self.dataset, self.collator, self.name, self.every_n_steps = dataset, collator, name, every_n_steps

    @torch.no_grad()
    def on_step_end(self, args, state, control, **kwargs):
        if (state.global_step % self.every_n_steps) == 0:
            batch = [self.dataset[random.randint(0, len(self.dataset)-1)] for _ in range(8)]
            out = self.collator(batch)
            valid = (out["labels"] != -100).sum().item()
            total = (out["attention_mask"] == 1).sum().item()
            print(f"\n[LabelStats:{self.name}] step={state.global_step} valid_ratio={valid/max(1,total):.4f}")


# -----------------------------
# 2.5) Google Drive バックアップ設定
# -----------------------------
# 学習完了後にGoogle Driveにバックアップを保存するための設定。
# ランタイム切断前に必要なファイルを保存することで、
# 再接続後にHugging Faceへのアップロードを継続できる。

# バックアップ先ディレクトリ
DRIVE_BACKUP_DIR = "/content/drive/MyDrive/lora_backup"

def save_to_google_drive(source_dir: str, backup_dir: str = DRIVE_BACKUP_DIR) -> str:
    """
    学習済みLoRAアダプターをGoogle Driveにバックアップする関数。

    Args:
        source_dir: バックアップ元ディレクトリ（OUT_LORA_DIR）
        backup_dir: バックアップ先ディレクトリ（Google Drive）

    Returns:
        str: バックアップ先パス
    """
    from pathlib import Path
    from datetime import datetime

    # タイムスタンプ付きバックアップフォルダ名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_name = f"agentbench_sft_{timestamp}"
    backup_path = os.path.join(backup_dir, experiment_name)

    os.makedirs(backup_path, exist_ok=True)

    # バックアップ対象ファイル
    backup_files = [
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",  # fallback
        "README.md",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    ]

    copied_files = []
    source_path = Path(source_dir)

    for filename in backup_files:
        src_file = source_path / filename
        if src_file.exists():
            dst_file = Path(backup_path) / filename
            shutil.copy2(src_file, dst_file)
            copied_files.append(filename)
            print(f"  ✓ {filename}")

    # 追加: *.json ファイルもコピー
    for json_file in source_path.glob("*.json"):
        if json_file.name not in [f for f in backup_files if f.endswith(".json")]:
            dst_file = Path(backup_path) / json_file.name
            shutil.copy2(json_file, dst_file)
            copied_files.append(json_file.name)
            print(f"  ✓ {json_file.name}")

    print(f"\n{'='*60}")
    print(f"📁 Google Driveバックアップ完了")
    print(f"   保存先: {backup_path}")
    print(f"   ファイル数: {len(copied_files)}")
    print(f"{'='*60}")
    print(f"\n💡 再接続後にHFアップロードを行う場合:")
    print(f"   1. ランタイム再接続")
    print(f"   2. Google Driveをマウント")
    print(f"   3. 以下のパスからファイルを読み込んでアップロード:")
    print(f"      {backup_path}")

    return backup_path


# -----------------------------
# 2.6) aggregation-MAX アップサンプリング
# -----------------------------
# ベースラインでaggregation-MAX問題の正解率が0%だった対策。
# DBBenchデータ内のMAX系クエリ（SELECT MAX(...) 等）を自動検出し、
# 指定倍率で複製することで学習時の出現頻度を引き上げる。

def upsample_aggregation_max(ds: Dataset, multiplier: int = 3) -> Dataset:
    """
    DBBenchデータ内のaggregation-MAX系クエリを検出してアップサンプリングする。
    assistant発話内に MAX( / max( / GREATEST( を含むサンプルを検出対象とする。

    Args:
        ds: 結合済みデータセット
        multiplier: アップサンプリング倍率（デフォルト3倍）

    Returns:
        Dataset: MAXクエリが増幅されたデータセット
    """
    import re
    max_pattern = re.compile(r'\b(MAX|max|GREATEST|greatest)\s*\(', re.IGNORECASE)

    max_indices = []
    for i in range(len(ds)):
        msgs = ds[i]["messages"]
        # assistant発話内にMAX関数パターンがあるか検査
        for msg in msgs:
            if msg.get("role") in ("assistant", "agent") and max_pattern.search(msg.get("content", "")):
                max_indices.append(i)
                break

    if not max_indices:
        print("[UPSAMPLE] MAX系クエリが検出されませんでした。スキップします。")
        return ds

    # 元のデータ + MAXサンプルを(multiplier - 1)回複製
    extra_indices = max_indices * (multiplier - 1)
    all_indices = list(range(len(ds))) + extra_indices

    ds_upsampled = ds.select(all_indices)
    print(f"[UPSAMPLE] aggregation-MAX: {len(max_indices)}件検出 → {multiplier}倍に増幅")
    print(f"[UPSAMPLE] データセットサイズ: {len(ds)} → {len(ds_upsampled)} (+{len(extra_indices)}件)")
    return ds_upsampled


# -----------------------------
# 2.7) Main
# -----------------------------
# main() が実行されると、ここまでの部品を使って学習が一気に進みます。

def load_single_dataset(dataset_id: str) -> Dataset:
    """
    単一データセットをHuggingFace Hubからロードする。
    標準ロードに失敗した場合、手動ダウンロード(JSONL)にフォールバック。

    Args:
        dataset_id: HuggingFace上のデータセットID

    Returns:
        Dataset: ロードされたデータセット
    """
    print(f"  [LOAD] {dataset_id}")
    try:
        ds = load_dataset(dataset_id, split="train", token=HF_TOKEN)
    except Exception as e:
        print(f"  [WARN] Standard load failed: {e}\n  Trying manual loading...")
        try:
            file_path = hf_hub_download(
                repo_id=dataset_id, filename="data.jsonl",
                repo_type="dataset", token=HF_TOKEN
            )
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            data_list = [json.loads(line) for line in lines if line.strip()]
            if not data_list:
                raise ValueError("Downloaded file is empty.")
            all_keys = set().union(*(d.keys() for d in data_list))
            for d in data_list:
                for k in all_keys:
                    if k not in d:
                        d[k] = None
            ds = Dataset.from_list(data_list)
            print(f"  [INFO] Manual load successful. {len(ds)} rows loaded.")
        except Exception as e2:
            print(f"  [ERROR] Manual load also failed: {e2}")
            raise e
    print(f"  [INFO] Loaded {len(ds)} rows. Columns: {ds.column_names}")
    return ds


def convert_single_dataset(ds: Dataset, dataset_id: str = "") -> Dataset:
    """
    単一データセットをmessages形式に変換する。
    role名の正規化（agent → assistant）も実施。

    Args:
        ds: 変換前のデータセット

    Returns:
        Dataset: messages形式に変換されたデータセット
    """
    dataset_id_lower = str(dataset_id).lower()

    if "messages" in ds.column_names:
        keep_cols = {"messages", "tools"}
        remove_cols = [c for c in ds.column_names if c not in keep_cols]
        if remove_cols:
            ds = ds.remove_columns(remove_cols)
        if ALF_REACT_CONVERSION and looks_like_alfworld_dataset(dataset_id, ds):
            print("  [INFO] Converting ALFWorld function-calling trajectories to text ReAct.")
            ds = ds.map(
                lambda ex: {"messages": convert_alfworld_messages_to_react(ex["messages"])},
                desc="Converting ALFWorld to ReAct"
            )
            if "tools" in ds.column_names:
                ds = ds.remove_columns(["tools"])
        else:
            ds = ds.map(
                lambda ex: {"messages": normalize_roles(ex["messages"])},
                desc="Normalizing roles"
            )
        has_tools = "tools" in ds.column_names
        print(f"  [INFO] messages column present. tools={'present' if has_tools else 'absent'}.")
    else:
        print("  [INFO] Converting from legacy format...")
        ds = ds.map(convert_to_messages_format, remove_columns=ds.column_names)

    if DB_CLEAN_PAREN_BUG and "dbbench" in dataset_id_lower and "messages" in ds.column_names:
        before = len(ds)
        ds = ds.filter(lambda ex: not has_db_paren_bug(ex["messages"]), desc="Removing DB paren bug rows")
        removed = before - len(ds)
        if removed:
            print(f"  [CLEAN] Removed {removed} DB rows with known SQL '))' bug.")
    return ds


def main():
    os.makedirs(OUT_LORA_DIR, exist_ok=True)

    if os.path.exists("/content/your_id"):
        shutil.rmtree("/content/your_id")

    # ============================================================
    # マルチデータセットのロードと結合
    # ============================================================
    print(f"[INFO] Loading {len(DATASET_IDS)} dataset(s)...")
    for i, ds_id in enumerate(DATASET_IDS):
        print(f"  [{i+1}/{len(DATASET_IDS)}] {ds_id}")

    all_datasets = []
    for ds_id in DATASET_IDS:
        ds = load_single_dataset(ds_id)
        ds = convert_single_dataset(ds, dataset_id=ds_id)
        all_datasets.append(ds)
        print(f"  → {len(ds)} samples converted.")

    # 複数データセットを結合
    from datasets import concatenate_datasets
    if len(all_datasets) == 1:
        ds_all = all_datasets[0]
    else:
        # toolsカラムの有無を統一（無い場合はNoneで埋める）
        has_tools_flags = ["tools" in ds.column_names for ds in all_datasets]
        if any(has_tools_flags) and not all(has_tools_flags):
            for i, ds in enumerate(all_datasets):
                if "tools" not in ds.column_names:
                    all_datasets[i] = ds.add_column("tools", [None] * len(ds))
        ds_all = concatenate_datasets(all_datasets)

    print(f"\n[INFO] Combined dataset: {len(ds_all)} total samples")
    print(f"[INFO] Columns: {ds_all.column_names}")

    # サンプル表示
    if len(ds_all) > 0:
        sample_messages = ds_all[0]['messages']
        print("First sample preview:")
        print(json.dumps(sample_messages[:3], indent=2, ensure_ascii=False))
        print(f"  ... ({len(sample_messages)} turns total)")

    # データ形式チェック（messagesがlistであること）
    ensure_openai_messages(ds_all)

    # 基本的なフィルタリング（アシスタントが存在する必要があり、最後のターンはアシスタント）
    # 学習できるサンプルだけ残す（assistantが空なら教師信号が無い）
    ds_all = ds_all.filter(lambda ex: has_any_nonempty_assistant_turn(ex["messages"]))
    ds_all = ds_all.filter(ends_with_nonempty_assistant)

    # aggregation-MAX アップサンプリング（オプション）
    if UPSAMPLE_AGG_MAX:
        print(f"[INFO] aggregation-MAX アップサンプリング有効 (倍率: {AGG_MAX_MULTIPLIER}x)")
        ds_all = upsample_aggregation_max(ds_all, multiplier=AGG_MAX_MULTIPLIER)
    else:
        print("[INFO] aggregation-MAX アップサンプリング無効")

    # train/val分割
    train_ds, val_ds = shuffle_split(ds_all, VAL_RATIO, SEED)

    print("[INFO] Loading base model:", BASE_MODEL_ID)

    # Unslothでベースモデルを読み込む
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        max_seq_length=MAX_SEQ_LEN,
        dtype=None,
        load_in_4bit=True,    # QLoRA: L4 (24GB) でも7B bf16は厳しいため4bit量子化
        use_exact_model_name=True,  # ← これでリマップを防止
    )

    # Cache chat template renders
    # ※現行Collatorはmessagesから直接境界計算するため、このキャッシュは
    #   Collator内部では参照されません。将来の拡張やデバッグ用に残しています。
    build_cache = make_text_cache_builder(tokenizer)
    train_ds = train_ds.map(build_cache, batched=True, num_proc=1, desc="Caching train")
    val_ds   = val_ds.map(build_cache,   batched=True, num_proc=1, desc="Caching val")

    # Attach LoRA
    # ここで「学習される部分（LoRAアダプタ）」をモデルに追加します。
    # 学習対象は LoRA のパラメータだけになり、ベースモデルの巨大な重みは固定されます。
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
    )

    # IMPORTANT FIX:
    # transformers TrainingArguments uses eval_strategy (NOT evaluation_strategy)
    # ※ここは重要：Transformersの引数名がバージョンで揺れることがあります。
    # 今回のバージョンでは eval_strategy を使います。
    training_args_kwargs = {}
    if NEFTUNE_NOISE_ALPHA > 0:
        training_args_kwargs["neftune_noise_alpha"] = NEFTUNE_NOISE_ALPHA
        print(f"[INFO] NEFTune enabled: alpha={NEFTUNE_NOISE_ALPHA}")

    args = TrainingArguments(
        output_dir=OUT_LORA_DIR,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        warmup_ratio=WARMUP_RATIO,
        lr_scheduler_type="cosine",
        weight_decay=WEIGHT_DECAY,

        logging_steps=LOGGING_STEPS,

        eval_strategy="steps",
        eval_steps=EVAL_STEPS,

        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,

        max_steps=MAX_STEPS,  # -1 => epoch-based

        bf16=True,             # L4はbf16ネイティブ対応
        fp16=False,

        push_to_hub=False,
        report_to="none",

        group_by_length=False,
        remove_unused_columns=False,
        **training_args_kwargs,
    )

    # assistant-only loss の collator を使う
    collator = AssistantOnlyCollatorCached(tokenizer=tokenizer, max_length=MAX_SEQ_LEN)

    # --- NaN対策：all-masked（教師トークン0）を除去して評価を安定化 ---
    print("[INFO] Checking all-masked samples before filtering...")
    count_all_masked(val_ds, collator, n=len(val_ds), seed=SEED)

    print("[INFO] Filtering train/val to remove all-masked samples...")
    train_ds = filter_has_supervision(train_ds, collator)
    val_ds   = filter_has_supervision(val_ds, collator)

    print("[INFO] New sizes:", "train =", len(train_ds), "val =", len(val_ds))
    print("[INFO] Checking all-masked samples after filtering...")
    count_all_masked(val_ds, collator, n=len(val_ds), seed=SEED)


    # Trainer（Transformersの標準学習ループ）
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
    )

    # 監視用コールバックを追加（学習が効いているかのヘルスチェック）
    trainer.add_callback(LabelStatsCallback(train_ds, collator, name="train", every_n_steps=LOGGING_STEPS))

    print("[INFO] Starting training...")
    trainer.train()

    # 学習後の保存：LoRAアダプタ＆tokenizer
    print("[INFO] Saving adapter & tokenizer...")
    model.save_pretrained(OUT_LORA_DIR)
    tokenizer.save_pretrained(OUT_LORA_DIR)
    print(f"[INFO] Done. Saved to {OUT_LORA_DIR}")

    # ============================================================
    # README.md 生成（学習パラメータから自動生成）
    # ============================================================
    print("\n" + "="*60)
    print("📝 README.md を生成中...")
    print("="*60)
    generate_readme()

    # ============================================================
    # LoRAマージ （アダプタをベースモデルに統合）
    # ============================================================
    print("\n" + "="*60)
    print("🔗 LoRAアダプタをベースモデルにマージ中...")
    print("="*60)
    merge_and_save_model(model, tokenizer)

    # ============================================================
    # HuggingFace アップロード
    # ============================================================
    if AUTO_UPLOAD_HF:
        print("\n" + "="*60)
        print("🚀 HuggingFace へアップロード中...")
        print("="*60)
        upload_to_hf()

    # ============================================================
    # Google Driveバックアップ（ランタイム切断前の保存）
    # ============================================================
    backup_path = None
    if SAVE_TO_DRIVE:
        print("\n" + "="*60)
        print("💾 Google Driveへバックアップ中...")
        print("="*60)
        backup_path = save_to_google_drive(OUT_LORA_DIR)

    # ============================================================
    # コンピューティングユニット節約: GPU自動解放
    # ============================================================
    release_gpu(model, trainer, backup_path)

def generate_readme() -> None:
    """
    学習パラメータから README.md（HFモデルカード）を自動生成し、
    OUT_LORA_DIR に保存する。

    Args:
        なし（グローバル変数 BASE_MODEL_ID, DATASET_IDS 等を参照）

    Returns:
        None
    """
    def _s(x, default=""):
        """値を安全に文字列化する補助関数。"""
        try:
            v = str(x)
            return v if v.strip() else default
        except Exception:
            return default

    def _fmt_lr(x) -> str:
        """学習率を指数表記に整形する補助関数。"""
        try:
            return f"{float(x):.0e}"
        except Exception:
            return _s(x, "")

    base_model_id = _s(BASE_MODEL_ID, "Qwen/Qwen2.5-7B-Instruct")
    datasets_yaml = "\n".join([f"- {ds_id}" for ds_id in DATASET_IDS])
    datasets_text = ", ".join(DATASET_IDS)
    max_seq_len = int(MAX_SEQ_LEN)
    epochs = int(NUM_TRAIN_EPOCHS)
    lr_str = _fmt_lr(LR)
    lora_r = int(LORA_R)
    lora_alpha = int(LORA_ALPHA)
    repo_license = os.environ.get("SFT_REPO_LICENSE", "apache-2.0")
    profile = _getenv("SFT_EXPERIMENT_PROFILE", "")
    adapter_repo = _getenv("HF_UPLOAD_REPO_ID", "your_id/your-repo")

    objective_text = (
        "This adapter is trained to improve **multi-turn agent task performance**\n"
        "on ALFWorld (household tasks) and DBBench (database operations)."
    )
    data_notes = []
    tag_lines = ["- lora", "- agent", "- tool-use", "- alfworld", "- dbbench"]

    if profile.startswith("hybrid_alf_react"):
        objective_text = (
            "This adapter is trained to improve **multi-turn agent task performance**\n"
            "on ALFWorld and DBBench under a shared **text ReAct-style** interaction format."
        )
        data_notes.append(
            "- ALFWorld trajectories were converted from function-calling format into text ReAct before training."
        )
        data_notes.append(
            "- DBBench trajectories were kept in text ReAct format and cleaned to remove known malformed SQL samples."
        )
        tag_lines.append("- react")
    else:
        data_notes.append(
            "- DBBench malformed SQL rows with the known trailing `))` bug were removed before training."
        )

    if UPSAMPLE_AGG_MAX:
        data_notes.append(
            f"- aggregation-MAX examples were upsampled by {AGG_MAX_MULTIPLIER}x during training."
        )

    method_text = "QLoRA (4-bit base model) + Unsloth"
    data_notes_text = "\n".join(data_notes)
    tags_yaml = "\n".join(tag_lines)

    readme_md = f"""---
base_model: {base_model_id}
datasets:
{datasets_yaml}
language:
- en
license: {repo_license}
library_name: peft
pipeline_tag: text-generation
tags:
{tags_yaml}
---

# AgentBench SFT LoRA Adapter

This repository provides a **LoRA adapter** fine-tuned from
**{base_model_id}** using **LoRA + Unsloth**.

This repository contains **LoRA adapter weights only**.
The base model must be loaded separately.

## Training Objective

{objective_text}

Loss is applied to **all assistant turns** in the multi-turn trajectory,
enabling the model to learn environment observation, action selection,
tool use, and recovery from errors.

## Training Configuration

- Base model: {base_model_id}
- Method: {method_text}
- Max sequence length: {max_seq_len}
- Epochs: {epochs}
- Learning rate: {lr_str}
- LoRA: r={lora_r}, alpha={lora_alpha}

## Usage

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

base = "{base_model_id}"
adapter = "{adapter_repo}"

tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(
    base,
    torch_dtype=torch.float16,
    device_map="auto",
)
model = PeftModel.from_pretrained(model, adapter)
```

## Sources & Terms (IMPORTANT)

Training data: {datasets_text}

Data notes:
{data_notes_text}

Dataset License: MIT License.
Compliance: Users must comply with the MIT license and the base model's original terms of use.
"""

    os.makedirs(OUT_LORA_DIR, exist_ok=True)
    readme_path = os.path.join(OUT_LORA_DIR, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_md)

    print(f"[INFO] README.md written to: {readme_path}")
    print("[INFO] Preview (first 10 lines):")
    for i, line in enumerate(readme_md.splitlines()[:10], start=1):
        print(f"  {i:02d}: {line}")


def merge_and_save_model(model, tokenizer) -> None:
    """
    学習済みLoRAアダプタをベースモデルにマージし、
    推論用の完全なモデルとして保存する。

    ※ Unsloth + QLoRA (load_in_4bit=True) 環境では、標準PEFTの
      merge_and_unload() は4bit量子化された重みを正しくdequantizeできない。
      代わりに Unsloth 専用の save_pretrained_merged() を使用する。

    Args:
        model: LoRAアダプタが適用された学習済みモデル（Unsloth FastLanguageModel）
        tokenizer: トークナイザー

    Returns:
        None
    """
    print(f"[INFO] Merging LoRA adapter into base model (Unsloth merged_16bit)...")
    os.makedirs(MERGED_MODEL_DIR, exist_ok=True)

    # Unsloth専用API: 4bit量子化モデルをfloat16にdequantize + LoRAマージして保存
    model.save_pretrained_merged(
        MERGED_MODEL_DIR,
        tokenizer,
        save_method="merged_16bit",  # float16で完全マージ
    )
    print("[INFO] Merge & save complete.")

    # README.md をマージ済みモデルディレクトリにもコピー
    readme_src = os.path.join(OUT_LORA_DIR, "README.md")
    if os.path.exists(readme_src):
        shutil.copy(readme_src, os.path.join(MERGED_MODEL_DIR, "README.md"))
        print("[INFO] README.md copied to merged model dir.")

    print(f"[INFO] Merged model saved to {MERGED_MODEL_DIR}")


def upload_to_hf() -> None:
    """
    マージ済みモデルをHuggingFace Hubにアップロードする。
    HF_UPLOAD_REPO_IDが未設定またはデフォルトのままの場合はスキップ。

    Args:
        なし（グローバル変数を参照）

    Returns:
        None
    """
    from huggingface_hub import HfApi, upload_folder

    repo_id = _getenv("HF_UPLOAD_REPO_ID", "")
    private = _getenv("HF_PRIVATE_REPO", "False").lower() in ("true", "1")

    # リポジトリIDが未設定 or デフォルトのままならスキップ
    if not repo_id or repo_id == "your-username/your-repo-name":
        print("[WARN] HF_UPLOAD_REPO_ID が未設定です。アップロードをスキップします。")
        print("       環境変数 HF_UPLOAD_REPO_ID にリポジトリIDを設定してください。")
        return

    if not os.path.exists(MERGED_MODEL_DIR):
        print(f"[ERROR] マージ済みモデルが見つかりません: {MERGED_MODEL_DIR}")
        return

    api = HfApi()

    print(f"[INFO] リポジトリ作成/確認: {repo_id}")
    api.create_repo(
        repo_id=repo_id,
        repo_type="model",
        exist_ok=True,
        private=private,
    )

    print(f"[INFO] アップロード中: {MERGED_MODEL_DIR} → {repo_id}")
    upload_folder(
        folder_path=MERGED_MODEL_DIR,
        repo_id=repo_id,
        repo_type="model",
        commit_message=f"Upload merged {BASE_MODEL_ID} + LoRA (auto)",
        token=HF_TOKEN,
    )

    print(f"\n✅ アップロード完了！")
    print(f"   https://huggingface.co/{repo_id}")


def release_gpu(model=None, trainer=None, backup_path=None) -> None:
    """
    GPU メモリを解放し、Colab環境ではコンピューティングユニットを節約するためにGPUを切断する。

    Args:
        model: 学習済みモデル（del対象）
        trainer: Trainerインスタンス（del対象）
        backup_path: Google Driveバックアップ先パス（表示用）

    Returns:
        None
    """
    if not AUTO_RELEASE_GPU:
        print("\n💡 GPU自動解放は無効です。手動で解放する場合:")
        print("   from google.colab import runtime; runtime.unassign()")
        return

    try:
        import gc
        if model is not None:
            del model
        if trainer is not None:
            del trainer
        gc.collect()
        torch.cuda.empty_cache()

        print("\n" + "="*60)
        print("💾 学習完了！チェックポイントを保存しました。")
        print(f"   保存先: {OUT_LORA_DIR}")
        if SAVE_TO_DRIVE and backup_path:
            print(f"   バックアップ: {backup_path}")
        print("="*60)

        try:
            from google.colab import runtime
            print("\n🔌 GPU解放中...（コンピューティングユニット節約）")
            print("   次のセッションでは Drive から読み込んで続行できます。")
            runtime.unassign()
        except ImportError:
            print("[INFO] Not running in Colab. Skipping GPU release.")
    except Exception as e:
        print(f"[WARNING] GPU release failed: {e}")


if __name__ == "__main__":
    main()


