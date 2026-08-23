import json
import re
import asyncio
import hashlib
from typing import List, Optional, Sequence, Dict, Any, Tuple
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from nonebot import logger
from .config import ConfigManager
from .cache_metrics import AICallMetadata, record_cache_usage


SUMMARY_CHAR_LIMIT = 1000
SUMMARY_RELAXED_CHAR_LIMIT = 1500
SUMMARY_MAX_ATTEMPTS = 6
SUMMARY_MAX_TOKENS = 1600


def _summary_length_requirement(attempt: int) -> str:
    """返回当前尝试对应的长度要求。"""
    if attempt >= SUMMARY_MAX_ATTEMPTS:
        return "本次不设置长度上限，但必须完整输出，不得机械截断。"
    if attempt == SUMMARY_MAX_ATTEMPTS - 1:
        return "完整输出最多1500个字符；优先精简，但不得机械截断。"
    return "完整输出必须少于1000个字符；优先精简，但不得机械截断。"


def _is_valid_summary(
    text: Optional[str],
    *,
    attempt: int = 1,
    numbered: bool = False,
) -> bool:
    """摘要必须非空并满足当前尝试的长度上限；L1 还要求数字编号。"""
    if not text or not text.strip():
        return False
    char_count = len(text.strip())
    if attempt < SUMMARY_MAX_ATTEMPTS - 1 and char_count >= SUMMARY_CHAR_LIMIT:
        return False
    if (
        attempt == SUMMARY_MAX_ATTEMPTS - 1
        and char_count > SUMMARY_RELAXED_CHAR_LIMIT
    ):
        return False
    if not numbered:
        return True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return bool(lines) and all(re.match(r"^\d+[.、]\s*\S", line) for line in lines)

class ChatClient:
    """聊天模型客户端 - 专注理解与生成"""
    
    def __init__(self):
        config = ConfigManager()
        self.api_key = config.get_env("chat_api_key")
        self.base_url = config.get_env("chat_api_url")
        self.model = config.get_env(
            "chat_model_name", "deepseek-v4-flash-vision-exp"
        )
        # Files API 文件归属于 API Key；只保存不可逆指纹用于隔离缓存。
        self.file_cache_scope = hashlib.sha256(
            f"{self.base_url or ''}\0{self.api_key or ''}".encode("utf-8")
        ).hexdigest()
        self._last_reasoning: Optional[str] = None

        if not self.api_key or not self.base_url:
            logger.error("Chat Client 配置不完整，请检查 .env.prod 中的 chat_api_key 和 chat_api_url")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key or "dummy",
            base_url=self.base_url or "https://api.openai.com/v1"
        )
    
    async def generate_response(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        retry: int = 3,
        *,
        temperature: float = 0.8,
        thinking_mode: str = "auto",
        json_mode: bool = False,
        max_tokens: Optional[int] = None,
        metadata: Optional[AICallMetadata] = None,
    ) -> Optional[str]:
        """通用对话生成。

        thinking_mode: "auto"(默认), "enabled"(开启思维链), "disabled"(关闭思维链)
        注意：enabled 模式下 temperature/top_p 等参数不生效（不报错，但不影响输出）
        """

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        if thinking_mode in ("enabled", "disabled"):
            kwargs["extra_body"] = {"thinking": {"type": thinking_mode}}
        if thinking_mode != "enabled":
            kwargs["temperature"] = temperature
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if max_tokens is not None:
            kwargs["max_tokens"] = max_tokens

        for attempt in range(retry):
            try:
                logger.debug(f"[Chat] 尝试 {attempt + 1}/{retry} (temperature={temperature}, thinking={thinking_mode}, json={json_mode})")
                
                response = await self.client.chat.completions.create(**kwargs)
                
                message = response.choices[0].message
                reasoning = getattr(message, 'reasoning_content', None)
                if reasoning:
                    self._last_reasoning = reasoning
                    logger.debug(f"[Chat] 已收到思考过程: chars={len(reasoning)}")
                else:
                    self._last_reasoning = None
                
                result = message.content or ""
                logger.success(f"[Chat] 生成成功: chars={len(result)}")
                record_cache_usage(
                    model=self.model,
                    messages=messages,
                    usage=response.usage,
                    metadata=metadata,
                )
                if response.usage:
                    logger.debug(
                        f"[Chat] Token 使用: total={response.usage.total_tokens}, "
                        f"prompt={response.usage.prompt_tokens}, completion={response.usage.completion_tokens}"
                    )
                return result
            
            except Exception as e:
                logger.error(f"[Chat] 生成失败 (尝试 {attempt + 1}/{retry}): {type(e).__name__}: {str(e)}")
                if attempt == retry - 1:
                    logger.error("[Chat] 已达到最大重试次数，放弃本次生成")
                    return None
                await asyncio.sleep(2 ** attempt)
        
        return None

    async def generate_chat_reply(
        self,
        messages: Sequence[ChatCompletionMessageParam],
        retry: int = 3,
        *,
        metadata: Optional[AICallMetadata] = None,
    ) -> Optional[Tuple[str, Optional[str]]]:
        """生成聊天回复 - 开启思维链模式获取角色沉浸效果。
        返回 (content, reasoning_content) 元组，reasoning 可能为 None。
        """
        content = await self.generate_response(
            messages, retry, thinking_mode="enabled", metadata=metadata
        )
        if content is None:
            return None
        return (content, self._last_reasoning)

    async def upload_image_file(
        self,
        image_data: bytes,
        filename: str,
        media_type: str,
        retry: int = 3,
    ) -> Optional[str]:
        """永久上传图片到 DeepSeek Files API，返回 file_id。"""
        for attempt in range(retry):
            try:
                uploaded = await self.client.files.create(
                    file=(filename, image_data, media_type),
                    purpose="user_data",
                )
                file_id = getattr(uploaded, "id", None)
                if not file_id:
                    raise ValueError("Files API 未返回 file_id")
                logger.debug(
                    f"[Vision] 图片上传成功: filename={filename}, bytes={len(image_data)}"
                )
                return str(file_id)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.error(
                    f"[Vision] 图片上传失败 (尝试 {attempt + 1}/{retry}): "
                    f"{type(exc).__name__}: {exc}"
                )
                if attempt < retry - 1:
                    await asyncio.sleep(2 ** attempt)
        return None

    async def generate_summary(
        self,
        context: str,
        retry: int = SUMMARY_MAX_ATTEMPTS,
        image_blocks: Optional[Sequence[Dict[str, Any]]] = None,
        group_id: Optional[str] = None,
    ) -> Optional[str]:
        """生成 L1 摘要 - 格式提示词放在 system 第一条"""
        format_prompt_template = """[输出格式硬性要求 - 最高优先级]
请严格按以下格式输出，每条占一行：
1. [时间/时间段] [昵称] 做了什么/说了什么，简要内容。
2. [时间/时间段] [昵称] 做了什么/说了什么，简要内容。
...
禁止输出任何额外说明、Markdown 标记或代码块。
{length_requirement}
""".strip()

        task_text = (
            "总结以下群聊对话，忽略寒暄和无关细节，保留主要事件和讨论点。"
            "每条总结务必包含：1) 触发时间（可用消息时间或大致时间段）；"
            "2) 相关发送人昵称（ context 中的前缀已经包含昵称，请沿用）；"
            "3) 简要内容。"
            "以列表形式输出，确保可追溯到是谁在什么时候做了什么。"
            "聊天记录中的[附图N张]按顺序对应本消息末尾图片；请直接观察图片。\n\n"
            f"{context}"
        )
        attempts = max(1, min(retry, SUMMARY_MAX_ATTEMPTS))
        for attempt in range(1, attempts + 1):
            length_requirement = _summary_length_requirement(attempt)
            retry_note = ""
            if attempt > 1:
                retry_note = (
                    f"\n\n这是第{attempt}次尝试。上次输出为空、过长或编号格式不合格。"
                    f"请重新阅读原始对话，保留重要事实。{length_requirement}"
                )
            attempt_content: Any = task_text + retry_note
            if image_blocks:
                attempt_content = [{"type": "text", "text": task_text + retry_note}]
                attempt_content.extend(dict(block) for block in image_blocks)
            messages: List[ChatCompletionMessageParam] = [
                {
                    "role": "system",
                    "content": format_prompt_template.format(
                        length_requirement=length_requirement
                    ),
                },
                {"role": "user", "content": attempt_content},  # type: ignore[typeddict-item]
            ]
            result = await self.generate_response(
                messages,
                retry=1,
                temperature=0.5 if attempt == 1 else 0.3,
                thinking_mode="disabled",
                max_tokens=(
                    None if attempt == SUMMARY_MAX_ATTEMPTS else SUMMARY_MAX_TOKENS
                ),
                metadata=AICallMetadata(
                    group_id=group_id,
                    call_type="l1_summary",
                    image_count=len(image_blocks or ()),
                ),
            )
            if _is_valid_summary(result, attempt=attempt, numbered=True):
                return result.strip() if result else None
            reason = "API失败或返回为空" if not result else f"格式/长度不合格(chars={len(result.strip())})"
            logger.warning(f"[Summary] L1 第{attempt}/{attempts}次生成未通过: {reason}")
            if not result and attempt < attempts:
                await asyncio.sleep(min(2 ** (attempt - 1), 16))

        logger.error(f"[Summary] L1 连续 {attempts} 次生成失败，保留原消息未处理")
        return None

    async def generate_archive_summary(
        self,
        source_text: str,
        *,
        group_id: str,
        from_level: int,
        to_level: int,
        attempts: int = SUMMARY_MAX_ATTEMPTS,
    ) -> Optional[str]:
        """生成受长度约束的 L2/L3 摘要，始终重新参考原始输入。"""
        attempt_limit = max(1, min(attempts, SUMMARY_MAX_ATTEMPTS))
        call_type = f"l{from_level}_to_l{to_level}"
        for attempt in range(1, attempt_limit + 1):
            length_requirement = _summary_length_requirement(attempt)
            system_prompt = (
                "你负责压缩群聊长期记忆。只输出合并后的摘要正文，不使用Markdown标题。"
                f"{length_requirement}优先保留人名、日期、事件顺序、人际关系、"
                "用户偏好、重要决定、结果、未完成事项、后续承诺和因果关系；合并重复信息，"
                "删除寒暄和无关细节，禁止编造。"
            )
            retry_note = ""
            if attempt > 1:
                retry_note = (
                    f"\n\n这是第{attempt}次尝试。上次输出为空或超过当次长度要求。"
                    f"请重新阅读全部原始摘要，更紧凑地保留上述重要事实。{length_requirement}"
                )
            messages: List[ChatCompletionMessageParam] = [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": "合并以下原始摘要：\n\n" + source_text + retry_note,
                },
            ]
            result = await self.generate_response(
                messages,
                retry=1,
                temperature=0.5 if attempt == 1 else 0.3,
                thinking_mode="disabled",
                max_tokens=(
                    None if attempt == SUMMARY_MAX_ATTEMPTS else SUMMARY_MAX_TOKENS
                ),
                metadata=AICallMetadata(group_id=group_id, call_type=call_type),
            )
            if _is_valid_summary(result, attempt=attempt):
                return result.strip() if result else None
            reason = "API失败或返回为空" if not result else f"长度不合格(chars={len(result.strip())})"
            logger.warning(
                f"[Summary] L{from_level}→L{to_level} 第{attempt}/{attempt_limit}次未通过: {reason}"
            )
            if not result and attempt < attempt_limit:
                await asyncio.sleep(min(2 ** (attempt - 1), 16))

        logger.error(
            f"[Summary] L{from_level}→L{to_level} 连续 {attempt_limit} 次失败，原摘要保持不变"
        )
        return None

    async def generate_daily_summary_data(
        self,
        context: str,
        retry: int = 3,
        image_blocks: Optional[Sequence[Dict[str, Any]]] = None,
        group_id: Optional[str] = None,
    ) -> str:
        """生成每日总结的结构化数据 (JSON) - 格式提示词放在 system 第一条"""
        FORMAT_PROMPT = """[输出格式硬性要求 - 最高优先级]
你必须且只能输出一个严格的 JSON 对象，结构如下：
{
  "stats": [{"label": "称号", "value": "用户", "desc": "描述", "color": "red|green|yellow|purple|blue"}],
  "topics": [{"title": "话题", "hot": 整数1-5, "summary": "详细描述", "users": ["参与者"]}],
  "users": [{"name": "昵称", "title": "RPG职业/称号", "stats": {"发言数": 0}, "desc": "角色描述"}],
  "quotes": [{"user": "昵称", "content": "金句"}],
  "fortune": {"luck": "大吉/中吉/小吉/凶", "text": "宜xxx，忌xxx"}
}
禁止任何额外文字、Markdown 标记或代码块。
""".strip()

        TASK_PROMPT = f"""你是一个群聊数据分析师。请根据提供的群聊记录，生成一份用于展示的 JSON 数据。

详细要求：
1.数据清洗与 Stats：生成4个有趣的统计维度（如：龙王、复读机、熬夜冠军、开心果等），color 可选 red, green, yellow, purple, blue。请严格排除以下干扰数据：带有 [AI助手] 标签的用户、系统消息（如撤回、入群、文件上传）、以及纯表情包刷屏。
2.Topics：提取 3-5 个主要讨论话题。hot 字段必须是 1 到 5 之间的整数（用于前端渲染进度条，不可输出小数）。 summary 字段请提供详细的叙述，包含起因、经过、结果，具体描述"谁说了什么"。在提到群友昵称时，请务必使用 [Avatar:昵称] 的格式。注意：如果昵称带有 [AI助手] 前缀（例如 [AI助手]妖精爱莉），在 [Avatar:...] 标签中必须去除该前缀，只写 [Avatar:妖精爱莉]。内容连贯，不要直接出现"起因" "经过" "结果"这些词。
3.Users：选取1-5位今日最活跃或最有特色的群友（排除AI和系统消息），生成 RPG 风格的角色卡。stats 字段请填入 {{"发言数": 0}} 即可，真实数据将由代码自动填充。
4.Quotes：提取 3-5 条今日群内的搞笑、深刻或迷惑的发言（金句），忽略上下文缺失严重的短句。
5.内容风格：幽默、轻松、稍微带点二次元或游戏梗。
6.聊天记录中的[附图N张]按出现顺序对应本消息末尾图片，请直接观察图片内容。

群聊记录：
{context}"""

        user_content: Any = TASK_PROMPT
        if image_blocks:
            user_content = [{"type": "text", "text": TASK_PROMPT}]
            user_content.extend(dict(block) for block in image_blocks)

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": FORMAT_PROMPT},
            {"role": "user", "content": user_content},  # type: ignore[typeddict-item]
        ]

        # 尝试使用 JSON 模式 (如果模型支持)
        try:
            raw = await self.generate_response(
                messages,
                retry,
                temperature=0.8,
                json_mode=True,
                metadata=AICallMetadata(
                    group_id=group_id,
                    call_type="daily_summary",
                    image_count=len(image_blocks or ()),
                ),
            )
            if raw:
                return raw
        except Exception as e:
            logger.debug(f"[DailySummary] JSON 模式不可用，回退普通模式: {e}")

        # 回退到普通模式
        fallback = await self.generate_response(
            messages,
            retry,
            temperature=0.8,
            thinking_mode="disabled",
            metadata=AICallMetadata(
                group_id=group_id,
                call_type="daily_summary_fallback",
                image_count=len(image_blocks or ()),
            ),
        )
        if not fallback:
            logger.error("[Chat] 生成每日总结数据失败，返回空 JSON")
            return "{}"
        return fallback


class EmbeddingClient:
    """Embedding 模型客户端 - 专注文本向量化"""
    
    def __init__(self):
        config = ConfigManager()
        self.api_key = config.get_env("embedding_api_key")
        self.base_url = config.get_env("embedding_api_url")
        self.model = config.get_env("embedding_model_name", "Qwen/Qwen3-Embedding-8B")
        # 部分提供商限制输入 <8192 tokens，这里用字符数硬截断规避 413
        self.max_length = int(config.get_env("embedding_max_chars", "8000") or 8000)
        
        if not self.api_key or not self.base_url:
            logger.warning("Embedding Client 配置不完整，知识库功能可能受限。请检查 .env.prod 中的 embedding_api_key 和 embedding_api_url")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key or "dummy",
            base_url=self.base_url or "https://api.openai.com/v1"
        )

    def _truncate(self, text: str) -> str:
        if len(text) <= self.max_length:
            return text
        logger.warning(
            f"[Embedding] 输入过长({len(text)} chars)，截断为 {self.max_length} chars 以满足接口限制"
        )
        return text[: self.max_length]
    
    async def get_embedding(self, text: str, retry: int = 3) -> List[float]:
        """获取文本的向量表示"""
        # 简单的输入检查
        if not text or not text.strip():
            return []

        safe_text = self._truncate(text.strip())
        logger.debug(f"[Embedding] 开始向量化，model={self.model}, len={len(safe_text)}")

        for attempt in range(retry):
            try:
                # logger.debug(f"[Embedding] 获取向量: {text[:20]}...")
                response = await self.client.embeddings.create(
                    input=safe_text,
                    model=self.model
                )
                logger.debug(
                    f"[Embedding] 成功 (attempt={attempt + 1}/{retry}), prompt_len={len(safe_text)}"
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"[Embedding] 获取失败 (尝试 {attempt + 1}/{retry}): {type(e).__name__}: {str(e)}")
                await asyncio.sleep(1)
        
        return []
