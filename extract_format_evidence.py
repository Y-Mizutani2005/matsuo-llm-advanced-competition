"""
ベンチマーク評価環境(AgentRuns.jsonl)のALFWorld出力フォーマットを抽出するスクリプト。

Args: なし
Returns: 標準出力にフォーマット比較結果を表示
"""
import json

with open("logs/baseline/AgentRuns.jsonl", "r", encoding="utf-8") as f:
    lines = f.readlines()

print("=" * 80)
print("[根拠1] ベンチマーク評価環境のフォーマット (AgentRuns.jsonl)")
print("=" * 80)

# --- index 49の history から ---
data0 = json.loads(lines[0])
hist = data0["output"]["history"]

print(f"\nindex: {data0['index']} / status: {data0['output']['status']}")
print(f"history全長: {len(hist)} ターン")

# タスク本番部分を探す（"Here is your task" を含むメッセージの次から）
task_idx = None
for i, msg in enumerate(hist):
    if "Here is your task" in msg.get("content", ""):
        task_idx = i
        break

if task_idx is None:
    # 直接タスク部分を探す
    task_idx = 2  # fallback

print(f"\n--- history[{task_idx}〜] タスク開始からの会話 ---\n")
for i in range(task_idx, min(task_idx + 8, len(hist))):
    msg = hist[i]
    role = msg["role"]
    content = msg["content"]
    if len(content) > 250:
        content = content[:250] + "..."
    print(f'  [{i}] role="{role}"')
    print(f"      {content}")
    print()

# --- logフィールド ---
print("-" * 80)
print("[根拠1b] 同index 49のlogフィールド（Round 5-9: 物体操作部分）")
print("-" * 80)

log = data0["output"]["result"]["log"]
for r in log[4:9]:
    rnd = r["round"]
    print(f"\n  Round {rnd}:")
    print(f"    モデル出力:  {r['output'][:140]}")
    print(f"    実行action:  {r['action']}")
    print(f"    環境の応答:  {r['observation'][:120]}")

print("\n\n" + "=" * 80)
print("[根拠2] ALFWorld v4 SFTデータのフォーマット（HuggingFace READMEより）")
print("=" * 80)

print("""
  SFTデータの各ターン:
  +--------------------------------------------------------------+
  | {"role": "assistant",                                        |
  |  "content": "Think: I should look for apple...",             |
  |  "tool_calls": [{                                            |
  |    "id": "call_1", "type": "function",                       |
  |    "function": {"name": "act",                               |
  |      "arguments": '{"action": "go to microwave 1"}'}         |
  |  }]}                                                         |
  | {"role": "tool",                                             |
  |  "tool_call_id": "call_1",                                   |
  |  "content": "The microwave 1 is open..."}                    |
  +--------------------------------------------------------------+

  ベンチマーク評価環境の各ターン:
  +--------------------------------------------------------------+
  | {"role": "agent",                                            |
  |  "content": "THOUGHT: ... ACTION: go to microwave 1"}        |
  | {"role": "user",                                             |
  |  "content": "The microwave 1 is open..."}                    |
  +--------------------------------------------------------------+
""")

print("=" * 80)
print("[差分まとめ]")
print("=" * 80)

print("""
  +------------------+----------------------+----------------------+
  |                  | SFTデータ(ALFWorld)  | ベンチマーク評価環境  |
  +------------------+----------------------+----------------------+
  | Agentのrole名    | "assistant"          | "agent"              |
  | アクション表現   | tool_calls JSON      | テキスト "ACTION: "  |
  | 思考表現         | content内 "Think: "  | content内 "THOUGHT:" |
  | 環境応答のrole   | "tool"               | "user"               |
  | tool_call_id     | あり                 | なし                 |
  +------------------+----------------------+----------------------+
""")
