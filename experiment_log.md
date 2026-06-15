# 実験ログ: Advanced Competition

試行錯誤の記録。各実験のスコアとHFリンクをここに記録する。

---

## Phase 1: ベースライン確立

### 実験 1-1: ベースライン（SFTなし）
- **モデル**: Qwen2.5-7B-Instruct
- **データ**: なし
- **設定**: SFTなし
- **HF**: 
- **スコア**: 
  - ALFWorld: 48.0%
  - DBBench: 51.0%
  - 合計: 99.0
- **メモ**: 初期地力は高いが、DBBenchのMAX系クエリは0.0%。DB task limitは7.3%、ALF invalid actionは44.0%。

---

## Phase 2: コア改善（7B + マルチDS + seq4096）

### 実験 2-1: 7B + マルチDS + role変換
- **モデル**: Qwen2.5-7B-Instruct
- **データ**: ALFWorld v4 + DBBench v4 + DBBench v1（計4,002件）
- **設定**: seq4096, BS=4(A100), GradAccum=2, LR=3e-6, epoch=2
- **HF**: 
- **スコア**: 
  - ALFWorld: 14.0%
  - DBBench: 41.5%
  - 合計: 55.5
- **メモ**: ベースライン比で ALF -34.0pt、DB -9.5pt、合計 -43.5pt。function-calling と text ReAct の混在で負の転移が発生。DB task limitは30.0%まで悪化し、v4ログでは45件中39件がSQL構文エラー1064。

---

## Phase 3: DB特化改善

### 実験 3-1: DBBench特化 + クレンジング + MAX増幅
- **モデル**: Qwen2.5-7B-Instruct
- **データ**: DBBench v1〜v4のみ（3,060件, ALFWorld排除）
- **設定**: seq4096, LR=1e-5, epoch=2, MAXクエリ3倍増幅, 括弧バグ6件除外
- **HF**: 
- **スコア**: 
  - ALFWorld: 28.0%
  - DBBench: 53.0%
  - 合計: 81.0
- **メモ**: Phase 2比で ALF +14.0pt、DB +11.5pt、合計 +25.5pt。ベースライン比では DB +2.0pt だが ALF -20.0pt。MAX正解率は0.0%→33.3%、DB task limitは30.0%→11.3%へ改善。ただし ALF invalid action は44.0%に戻り、総合ではベースライン未満。

---

## Phase 4: 低侵襲DB-only

### 実験 4-1: DB-low-impact
- **モデル**: Qwen2.5-7B-Instruct
- **データ**: DBBench v1〜v4のみ
- **設定**: low-impact SFT, 1epoch, low LR, MAX upsamplingなし
- **HF**: 
- **スコア**:
  - ALFWorld: 56.0%
  - DBBench: 52.405%
  - 合計: 108.405
  - 提出スコア: 4.1694
- **メモ**: ベースライン比で ALF +8.0pt、DB +1.405pt、合計 +9.405pt。DB task limitは7.3%→2.67%、ALF invalid actionは44.0%→20.0%まで改善。MAXは16.7%でPhase 3の33.3%には届かないが、総合バランスは現状ベスト。

---

## Phase 5: Hybrid ReAct混合

### 実験 5-1: hybrid_alf_react（MAXなし）
- **モデル**: Qwen2.5-7B-Instruct
- **データ**: DBBench v1〜v4 + ALFWorld v5（ALFはtext ReActへ変換）
- **設定**: hybrid SFT, 1epoch, LR=1e-6, MAX upsamplingなし
- **HF**: 
- **スコア**:
  - ALFWorld: 64.0%
  - DBBench: 48.806%
  - 合計: 112.806
  - 提出スコア: 4.3387
- **メモ**: DB-low-impact比で ALF +8.0pt、DB -3.599pt、合計 +4.401pt。ベースライン比では ALF +16.0pt、DB -2.194pt だが総合は大幅改善。ALF invalid actionは10.0%まで低下し、今回の実験群では総合トップ。DB task limitは4.67%で許容圏だが、DBカテゴリ精度はlow-impactよりやや悪化。

### 実験 5-2: hybrid_alf_react_max2
- **モデル**: Qwen2.5-7B-Instruct
- **データ**: DBBench v1〜v4 + ALFWorld v5（ALFはtext ReActへ変換）
- **設定**: hybrid SFT, 1epoch, LR=1e-6, aggregation-MAX upsampling x2
- **HF**: 
- **スコア**:
  - ALFWorld: 64.0%
  - DBBench: 48.715%
  - 合計: 112.715
  - 提出スコア: 4.3352
- **メモ**: `hybrid_alf_react` 比で ALF は同値、DB は -0.091pt、合計は -0.091pt。MAX upsampling を入れても aggregation-MAX は 16.7% で改善せず、全体ではほぼ横ばいか微悪化。提出候補としては `MAXなし hybrid` の方がわずかに上。

---

## 改善サマリ

- **ベースライン合計**: 99.0
- **Phase 2 合計**: 55.5
- **Phase 3 合計**: 81.0
- **Phase 4 合計**: 108.405
- **Phase 5 合計**: 112.806
- **最良DBスコア**: 53.0%（Phase 3）
- **最良ALFスコア**: 64.0%（Phase 5: hybrid_alf_react）
- **最良総合スコア**: 112.806（Phase 5: hybrid_alf_react）
- **現時点の結論**: `ALF function-calling → text ReAct 変換` を入れた hybrid が、DB-only low-impact を総合で上回った。MAX x2 を足しても改善は出なかったため、現時点の第一提出候補は `hybrid_alf_react`。

---
