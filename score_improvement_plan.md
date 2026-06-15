# Advanced Competition スコア改善計画

## 目的

AgentBench（DB Bench + ALFWorld）のスコアを改善するため、SFT学習コードを改修し、
データ戦略・モデル選択・ハイパーパラメータを最適化する。

---

## 戦略の柱

### 1. ベースモデル: Qwen2.5-7B-Instruct

| 根拠 | 詳細 |
|------|------|
| 上位モデルの傾向 | スコアボード上位の参加者が全員7Bを採用 |
| 容量 vs 性能 | 4Bよりエージェント的推論に強く、L4/A100で学習可能 |

### 2. マルチデータセット統合（計4,002件）

| データセット | 件数 | 選定理由 |
|-------------|------|---------|
| ALFWorld v4 | 2,502 | 最大かつ難易度バランスが良い |
| DBBench v4 | 1,200 | テーブル多様性最高（400種）、最多利用（537モデル） |
| DBBench v1 | 300 | テーブル多様性71%で補完、v4と重複しにくい |

### 3. シーケンス長拡大: 2048 → 4096

- 推論環境の `max-model-len` は 8192
- 長いトラジェクトリ（エラー回復・複数ステップ）を切り捨てない

### 4. DBBench role名変換（バグ修正）

- DBBenchデータは `role: "agent"` を使用
- 標準コードのcollatorは `"assistant"` を前提 → **変換しないとlossが0になる**
- `normalize_roles()` 関数で自動変換

---

## 改修内容まとめ

### ハイパーパラメータ

| パラメータ | 変更前 | 変更後 |
|-----------|--------|--------|
| ベースモデル | Qwen3-4B-Instruct-2507 | **Qwen2.5-7B-Instruct** |
| MAX_SEQ_LEN | 2048 | **4096** |
| バッチサイズ | 2 | **4**（A100向け） |
| 勾配累積 | 4 | **2** |
| 実効バッチサイズ | 8 | 8（同一） |
| 学習率 | 2e-6 | **3e-6** |
| エポック数 | 2 | 2（変更なし） |

### コード変更

| 変更 | 関数/変数 | 内容 |
|------|----------|------|
| マルチDS対応 | `DATASET_IDS`, `load_single_dataset()`, `convert_single_dataset()` | カンマ区切りで複数DS指定、個別ロード→結合 |
| role変換 | `normalize_roles()` | `agent` → `assistant` |
| README自動生成 | `generate_readme()` | 学習パラメータから自動生成 |
| モデルマージ | `merge_and_save_model()` | LoRAをベースモデルに統合 |
| HFアップロード | `upload_to_hf()` | マージ済みモデルを自動アップロード |
| GPU解放 | `release_gpu()` | メモリ解放＋Colabユニット節約 |

### 自動化フロー（main()）

```
main()
  ├─ データロード・結合（4,002件）
  ├─ フィルタリング・train/val分割
  ├─ モデルロード + LoRA適用
  ├─ SFT学習
  ├─ アダプタ保存
  ├─ 📝 generate_readme()
  ├─ 🔗 merge_and_save_model()
  ├─ 🚀 upload_to_hf()
  ├─ 💾 save_to_google_drive()
  └─ 🔌 release_gpu()
```

---

## 実行手順

1. `standard_code_sft_v2.py` を Colab にアップロード
2. `SFT_EXPERIMENT_PROFILE` を `db_low_impact` / `db_max2` / `hybrid_alf_react` のいずれかに設定
3. 必要に応じて `HF_UPLOAD_USER` と `HF_PRIVATE_REPO` を環境変数で設定
4. ランタイムを GPU に設定
5. セルを上から順に実行し、学習、マージ、アップロードの結果を確認
6. Omnicampus からサブミット

---

## 今後の拡張候補

- [ ] エラー回復パターンのデータ拡張
- [ ] ループ防止戦略の導入（上位モデル名に "loop breaker" が多い）
- [ ] DPO学習の追加
- [ ] 蒸留データの生成（ホワイトリスト掲載モデル利用）
