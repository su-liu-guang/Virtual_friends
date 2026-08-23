"""手工归档积压的 L1→L2→L3 摘要，不包含异常摘要扫描。"""

import asyncio
from pathlib import Path
import re
import sys
from typing import List, Optional, Sequence, Tuple

import aiosqlite
from openai import AsyncOpenAI


ROOT_PATH = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT_PATH / "data" / "Virtual_friends" / "memory.db"
ENV_PATH = ROOT_PATH / ".env.prod"
SUMMARY_CHAR_LIMIT = 1000
SUMMARY_RELAXED_CHAR_LIMIT = 1500
SUMMARY_MAX_ATTEMPTS = 6
SUMMARY_MAX_TOKENS = 1600
ARCHIVE_LEVELS = ((1, 2, 20), (2, 3, 10))


def load_dotenv(path: Path) -> dict:
    env = {}
    if not path.exists():
        print(f"[WARN] 未找到 {path}")
        return env
    with path.open(encoding="utf-8") as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            match = re.match(r"^(\w+)\s*=\s*(.+)$", line)
            if match:
                env[match.group(1)] = match.group(2).strip().strip('"').strip("'")
    return env


async def generate_bounded_summary(
    client: AsyncOpenAI,
    model: str,
    source_text: str,
    from_level: int,
    to_level: int,
) -> Optional[str]:
    for attempt in range(1, SUMMARY_MAX_ATTEMPTS + 1):
        if attempt == SUMMARY_MAX_ATTEMPTS:
            length_requirement = "本次不设置长度上限，但必须完整输出，不得机械截断。"
        elif attempt == SUMMARY_MAX_ATTEMPTS - 1:
            length_requirement = "完整输出最多1500个字符；优先精简，但不得机械截断。"
        else:
            length_requirement = "完整输出必须少于1000个字符；优先精简，但不得机械截断。"
        system_prompt = (
            "你负责压缩群聊长期记忆。只输出摘要正文。"
            f"{length_requirement}优先保留人名、日期、事件顺序、人际关系、用户偏好、"
            "重要决定、结果、未完成事项、后续承诺和因果关系；合并重复信息，"
            "删除寒暄，禁止编造。"
        )
        retry_note = ""
        if attempt > 1:
            retry_note = (
                f"\n\n这是第{attempt}次尝试。上次输出为空或超过当次长度要求，"
                f"请重新阅读全部原始摘要并更紧凑地保留重要事实。{length_requirement}"
            )
        try:
            request_kwargs = dict(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": "合并以下原始摘要：\n\n" + source_text + retry_note,
                    },
                ],
                temperature=0.5 if attempt == 1 else 0.3,
                extra_body={"thinking": {"type": "disabled"}},
            )
            if attempt < SUMMARY_MAX_ATTEMPTS:
                request_kwargs["max_tokens"] = SUMMARY_MAX_TOKENS
            response = await client.chat.completions.create(**request_kwargs)
            result = (response.choices[0].message.content or "").strip()
            is_valid_length = (
                attempt == SUMMARY_MAX_ATTEMPTS
                or (
                    attempt == SUMMARY_MAX_ATTEMPTS - 1
                    and len(result) <= SUMMARY_RELAXED_CHAR_LIMIT
                )
                or (
                    attempt < SUMMARY_MAX_ATTEMPTS - 1
                    and len(result) < SUMMARY_CHAR_LIMIT
                )
            )
            if result and is_valid_length:
                usage = response.usage
                print(
                    f"  [API] L{from_level}→L{to_level} attempt={attempt} "
                    f"chars={len(result)} prompt_tokens={getattr(usage, 'prompt_tokens', None)}"
                )
                return result
            print(
                f"  [WARN] L{from_level}→L{to_level} attempt={attempt} "
                f"内容不合格 chars={len(result)}"
            )
        except Exception as exc:
            print(
                f"  [WARN] L{from_level}→L{to_level} attempt={attempt} "
                f"API失败: {type(exc).__name__}"
            )
            if attempt < SUMMARY_MAX_ATTEMPTS:
                await asyncio.sleep(min(2 ** (attempt - 1), 16))
    return None


async def archive_level(
    db: aiosqlite.Connection,
    client: AsyncOpenAI,
    model: str,
    group_id: str,
    from_level: int,
    to_level: int,
    batch_size: int,
) -> int:
    created = 0
    while True:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM summaries "
            "WHERE group_id=? AND level=? AND is_archived=0",
            (group_id, from_level),
        )
        count = (await cursor.fetchone())[0]
        if count < batch_size:
            return created

        cursor = await db.execute(
            "SELECT id, content, time_range FROM summaries "
            "WHERE group_id=? AND level=? AND is_archived=0 "
            "ORDER BY created_at, id LIMIT ?",
            (group_id, from_level, batch_size),
        )
        rows: Sequence[Tuple[int, str, str]] = await cursor.fetchall()
        if len(rows) != batch_size:
            return created

        merged = await generate_bounded_summary(
            client,
            model,
            "\n\n".join(row[1] for row in rows),
            from_level,
            to_level,
        )
        if not merged:
            print(f"  [ERROR] L{from_level}→L{to_level} 连续6次失败，停止本层")
            return created

        ids = [row[0] for row in rows]
        time_range = f"{rows[0][2].split('-')[0]}-{rows[-1][2].split('-')[-1]}"
        placeholders = ",".join("?" for _ in ids)
        try:
            await db.execute("BEGIN IMMEDIATE")
            await db.execute(
                "INSERT INTO summaries "
                "(group_id, level, content, time_range, is_archived) "
                "VALUES (?, ?, ?, ?, 0)",
                (group_id, to_level, merged, time_range),
            )
            cursor = await db.execute(
                f"UPDATE summaries SET is_archived=1 "
                f"WHERE id IN ({placeholders}) AND is_archived=0",
                ids,
            )
            if cursor.rowcount != len(ids):
                raise RuntimeError(
                    f"归档并发冲突: expected={len(ids)}, updated={cursor.rowcount}"
                )
            await db.commit()
        except Exception:
            await db.rollback()
            raise

        created += 1
        print(
            f"  [OK] L{from_level}→L{to_level} #{created}: "
            f"{len(rows)}条→1条({len(merged)}字)"
        )


async def main() -> None:
    env = load_dotenv(ENV_PATH)
    api_key = env.get("chat_api_key", "")
    base_url = env.get("chat_api_url", "https://api.deepseek.com/v1")
    model = env.get("chat_model_name", "deepseek-v4-flash-vision-exp")
    if not api_key:
        print("[FATAL] chat_api_key 未设置")
        return

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    async with aiosqlite.connect(str(DB_PATH)) as db:
        if len(sys.argv) > 1:
            target_groups: List[str] = sys.argv[1:]
        else:
            cursor = await db.execute(
                "SELECT DISTINCT group_id FROM summaries "
                "WHERE is_archived=0 AND level IN (1, 2) ORDER BY group_id"
            )
            target_groups = [row[0] for row in await cursor.fetchall()]

        print(f"[INFO] model={model}, groups={len(target_groups)}, db={DB_PATH}")
        for group_id in target_groups:
            print(f"[GROUP] {group_id}")
            for from_level, to_level, batch_size in ARCHIVE_LEVELS:
                created = await archive_level(
                    db,
                    client,
                    model,
                    group_id,
                    from_level,
                    to_level,
                    batch_size,
                )
                print(f"  [DONE] L{from_level}→L{to_level}: created={created}")


if __name__ == "__main__":
    asyncio.run(main())
