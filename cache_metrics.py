"""DeepSeek 提示词缓存用量监控。"""

from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Optional, Sequence

from nonebot import logger
from openai.types.chat import ChatCompletionMessageParam


CACHE_METRIC_LEVEL = "CACHE_METRIC"
CACHE_METRIC_PATH = Path("data/Virtual_friends/logs/cache_metrics.jsonl")


@dataclass(frozen=True)
class AICallMetadata:
    """不会包含对话正文的 AI 调用标识。"""

    group_id: Optional[str] = None
    call_type: str = "unknown"
    image_count: int = 0


_sink_id: Optional[int] = None


def _ensure_metric_level() -> None:
    try:
        logger.level(CACHE_METRIC_LEVEL)
    except ValueError:
        logger.level(CACHE_METRIC_LEVEL, no=1, color="", icon="")


def init_cache_metrics() -> None:
    """创建独立 JSONL sink；写入由 Loguru 的队列线程完成。"""
    global _sink_id
    if _sink_id is not None:
        return

    CACHE_METRIC_PATH.parent.mkdir(parents=True, exist_ok=True)
    _ensure_metric_level()

    _sink_id = logger.add(
        CACHE_METRIC_PATH,
        level=CACHE_METRIC_LEVEL,
        format="{message}",
        filter=lambda record: bool(record["extra"].get("cache_metric")),
        rotation="10 MB",
        retention="30 days",
        compression="gz",
        enqueue=True,
        encoding="utf-8",
    )
    logger.info(f"[Cache] 缓存监控已写入 {CACHE_METRIC_PATH}")


def shutdown_cache_metrics() -> None:
    """移除 sink 并等待异步队列写完。"""
    global _sink_id
    if _sink_id is None:
        return
    logger.remove(_sink_id)
    _sink_id = None


def _get_usage_value(usage: Any, field: str) -> Optional[int]:
    if usage is None:
        return None
    value = getattr(usage, field, None)
    if value is None:
        extra = getattr(usage, "model_extra", None)
        if isinstance(extra, dict):
            value = extra.get(field)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _count_images(messages: Sequence[ChatCompletionMessageParam]) -> int:
    total = 0
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        total += sum(
            1
            for block in content
            if isinstance(block, dict) and block.get("type") in {"file", "image_url"}
        )
    return total


def _system_hash(messages: Sequence[ChatCompletionMessageParam]) -> str:
    system_messages = [
        message.get("content", "")
        for message in messages
        if message.get("role") == "system"
    ]
    payload = json.dumps(system_messages, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def record_cache_usage(
    *,
    model: str,
    messages: Sequence[ChatCompletionMessageParam],
    usage: Any,
    metadata: Optional[AICallMetadata],
) -> None:
    """将一次 Chat Completions 用量写入专用文件与精简终端日志。"""
    _ensure_metric_level()
    prompt_tokens = _get_usage_value(usage, "prompt_tokens")
    completion_tokens = _get_usage_value(usage, "completion_tokens")
    total_tokens = _get_usage_value(usage, "total_tokens")
    hit_tokens = _get_usage_value(usage, "prompt_cache_hit_tokens")
    miss_tokens = _get_usage_value(usage, "prompt_cache_miss_tokens")

    # 同时兼容 OpenAI 风格 prompt_tokens_details.cached_tokens。
    details = getattr(usage, "prompt_tokens_details", None) if usage is not None else None
    if hit_tokens is None and details is not None:
        hit_tokens = _get_usage_value(details, "cached_tokens")
    if miss_tokens is None and prompt_tokens is not None and hit_tokens is not None:
        miss_tokens = max(0, prompt_tokens - hit_tokens)

    denominator = None
    if hit_tokens is not None and miss_tokens is not None:
        denominator = hit_tokens + miss_tokens
    hit_rate = round(hit_tokens / denominator, 6) if denominator else None

    call = metadata or AICallMetadata()
    image_count = call.image_count or _count_images(messages)
    system_hash = _system_hash(messages)
    metric = {
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "group_id": call.group_id,
        "call_type": call.call_type,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "cache_hit_tokens": hit_tokens,
        "cache_miss_tokens": miss_tokens,
        "cache_hit_rate": hit_rate,
        "message_count": len(messages),
        "image_count": image_count,
        "system_hash": system_hash,
    }

    logger.bind(cache_metric=True).log(
        CACHE_METRIC_LEVEL,
        json.dumps(metric, ensure_ascii=False, separators=(",", ":")),
    )
    rate_text = f"{hit_rate:.2%}" if hit_rate is not None else "unknown"
    logger.info(
        f"[Cache] group={call.group_id or '-'} type={call.call_type} "
        f"hit={hit_tokens} miss={miss_tokens} rate={rate_text} "
        f"system={system_hash}"
    )
