# Advanced Competition 開発ドキュメント

## コンセプト

DBBench と ALFWorld の 2 タスクを、単一モデルの hybrid SFT で同時に改善する。

## 独自性のある部分

DBBench と ALFWorld は、どちらも agent 的な tool-use を扱う一方で、会話形式と action 表現が異なる。そこで ALFWorld 側の function-calling 形式 trajectory を text ReAct 形式に変換し、DBBench と近い形式に寄せたうえで hybrid SFT を行った。

## モデルと学習方法

- ベースモデル: `Qwen2.5-7B-Instruct`
- 追加学習: SFT with LoRA / QLoRA
- 量子化: 4bit
- max sequence length: 4096
- epoch: 1
- learning rate: `1e-6`
- LoRA r: 64
- LoRA alpha: 128
- gradient accumulation: 8

## 使った学習データと工夫

- DBBench: `dbbench_sft_dataset_react` v1-v4
- ALFWorld: `sft_alfworld_trajectory_dataset_v5`
- DBBench の `agent` role を `assistant` に正規化
- SQL 末尾に不要な `))` を含む不良サンプルを除去
- ALFWorld の function-calling action を text ReAct 形式へ変換

## データ形式統一処理

assistant message 内の `tool_calls` から action を抽出し、本文中の thought を整形したうえで `THOUGHT:` と `ACTION:` を持つ text ReAct 形式へ再構成した。

変換前:

```text
assistant:
Think: I should open the fridge first.
tool_calls: {"action": "open fridge 1"}
```

変換後:

```text
assistant:
THOUGHT: I should open the fridge first.
ACTION: open fridge 1
```

tool role の出力は次ターンの observation として user 側へ寄せ、DBBench と近い対話構造になるよう統一した。

## 試行錯誤

- ALFWorld と DBBench をそのまま混ぜると、形式の違いで性能が大きく崩れた。
- DBBench のみを強く学習すると DBBench は伸びる一方で、ALFWorld が悪化した。
- 最終的には「形式をそろえて混ぜる」「学習を強くしすぎない」という方針が最もバランス良かった。

## 結果

最良 profile は `hybrid_alf_react`。baseline の combined score `99.0` に対して、最良 combined score は `112.806` だった。
