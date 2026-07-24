"""一次性归档脚本：处理积压的 L1→L2→L3 摘要归档。

直接操作 SQLite + OpenAI API，不依赖 NoneBot。
"""
import asyncio
import sys
import os
import re
import aiosqlite
from pathlib import Path
from openai import AsyncOpenAI

DB_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "Virtual_friends" / "memory.db"
ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env.prod"


def load_dotenv(path: str) -> dict:
    env = {}
    if not os.path.exists(path):
        print(f"[WARN] 未找到 {path}")
        return env
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r'^(\w+)\s*=\s*(.+)$', line)
            if m:
                env[m.group(1)] = m.group(2).strip().strip('"').strip("'")
    return env


async def main():
    env = load_dotenv(str(ENV_PATH))
    api_key = env.get("chat_api_key", "")
    base_url = env.get("chat_api_url", "https://api.deepseek.com/v1")
    model = env.get("chat_model_name", "deepseek-chat")

    if not api_key:
        print("[FATAL] chat_api_key 未设置")
        return

    print(f"[INFO] API: {base_url}, model={model}, db={DB_PATH}")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    if len(sys.argv) > 1:
        target_groups = sys.argv[1:]
    else:
        async with aiosqlite.connect(str(DB_PATH)) as db:
            cursor = await db.execute(
                "SELECT DISTINCT group_id FROM summaries WHERE level=1 AND is_archived=0"
            )
            rows = await cursor.fetchall()
            target_groups = [r[0] for r in rows]

    for group_id in target_groups:
        print(f"\n{'='*50}")
        print(f"归档: group={group_id}")

        async with aiosqlite.connect(str(DB_PATH)) as db:
            # --- 查询初始状态 ---
            async def count_level(level):
                c = await db.execute(
                    "SELECT COUNT(*) FROM summaries WHERE group_id=? AND level=? AND is_archived=0",
                    (group_id, level),
                )
                return (await c.fetchone())[0]

            l1 = await count_level(1)
            l2 = await count_level(2)
            l3 = await count_level(3)
            print(f"初始: L1={l1}, L2={l2}, L3={l3}")

            # ======== 阶段1: L1 → L2 ========
            l2_created = 0
            while True:
                l1 = await count_level(1)
                if l1 < 80:
                    break

                cur = await db.execute(
                    "SELECT id, content, time_range FROM summaries "
                    "WHERE group_id=? AND level=1 AND is_archived=0 "
                    "ORDER BY created_at LIMIT 80",
                    (group_id,),
                )
                rows = await cur.fetchall()
                if not rows:
                    break

                ids = [r[0] for r in rows]
                combined = "\n\n".join([r[1] for r in rows])
                prompt = f"将以下多条摘要合并为一条更高层次的概括,保留关键信息:\n\n{combined}"

                try:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.5,
                    )
                    merged = resp.choices[0].message.content
                except Exception as e:
                    print(f"  [ERROR] API: {e}")
                    break

                if not merged:
                    print("  [ERROR] 合并结果为空")
                    break

                start_t = rows[0][2].split("-")[0]
                end_t = rows[-1][2].split("-")[-1]
                time_range = f"{start_t}-{end_t}"

                await db.execute(
                    "INSERT INTO summaries (group_id, level, content, time_range, is_archived) VALUES (?,2,?,?,0)",
                    (group_id, merged, time_range),
                )
                placeholders = ",".join(["?"] * len(ids))
                await db.execute(
                    f"UPDATE summaries SET is_archived=1 WHERE id IN ({placeholders})",
                    ids,
                )
                await db.commit()

                l2_created += 1
                l1_remain = await count_level(1)
                print(f"  L1→L2 #{l2_created}: {len(rows)}条→1条L2({len(merged)}字), 剩余L1={l1_remain}")

            print(f"[阶段1] 创建 {l2_created} 条 L2")

            # ======== 阶段2: L2 → L3 ========
            l2 = await count_level(2)
            l3_created = 0
            while l2 >= 30:
                cur = await db.execute(
                    "SELECT id, content, time_range FROM summaries "
                    "WHERE group_id=? AND level=2 AND is_archived=0 "
                    "ORDER BY created_at LIMIT 30",
                    (group_id,),
                )
                rows = await cur.fetchall()
                if not rows:
                    break

                ids = [r[0] for r in rows]
                combined = "\n\n".join([r[1] for r in rows])
                prompt = f"将以下多条摘要合并为一条更高层次的概括,保留关键信息:\n\n{combined}"

                try:
                    resp = await client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.5,
                    )
                    merged = resp.choices[0].message.content
                except Exception as e:
                    print(f"  [ERROR] API: {e}")
                    break

                if not merged:
                    print("  [ERROR] 合并结果为空")
                    break

                start_t = rows[0][2].split("-")[0]
                end_t = rows[-1][2].split("-")[-1]
                time_range = f"{start_t}-{end_t}"

                await db.execute(
                    "INSERT INTO summaries (group_id, level, content, time_range, is_archived) VALUES (?,3,?,?,0)",
                    (group_id, merged, time_range),
                )
                placeholders = ",".join(["?"] * len(ids))
                await db.execute(
                    f"UPDATE summaries SET is_archived=1 WHERE id IN ({placeholders})",
                    ids,
                )
                await db.commit()

                l3_created += 1
                l2 = await count_level(2)
                print(f"  L2→L3 #{l3_created}: {len(rows)}条→1条L3, 剩余L2={l2}")

            print(f"[阶段2] 创建 {l3_created} 条 L3")

            l1 = await count_level(1)
            l2 = await count_level(2)
            l3 = await count_level(3)
            print(f"最终: L1={l1}, L2={l2}, L3={l3}")

    print(f"\n{'='*50}")
    print("归档完成")


if __name__ == "__main__":
    asyncio.run(main())
