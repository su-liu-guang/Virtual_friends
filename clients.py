import json
import re
import asyncio
from typing import List, Optional, Sequence, Dict, Any
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from nonebot import logger
from .config import ConfigManager

class VisionClient:
    """视觉模型客户端 - 专注图像转文本"""
    
    def __init__(self):
        config = ConfigManager()
        self.api_key = config.get_env("vision_api_key")
        self.base_url = config.get_env("vision_api_url")
        self.model = config.get_env("vision_model_name", "gpt-4o-mini")
        
        
        if not self.api_key or not self.base_url:
            logger.error("Vision Client 配置不完整，请检查 .env.dev 文件中的 vision_api_key 和 vision_api_url")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key or "dummy",
            base_url=self.base_url or "https://api.openai.com/v1"
        )
    
    async def recognize_image(self, image_url: str, retry: int = 3, is_sticker: bool = False) -> str:
        """识别图片内容"""
        logger.debug(f"[Vision] 开始识别图片: {image_url[:50]}... (表情包: {is_sticker})")
        
        prompt = "请详细描述这张图片的内容,包括场景、物体、文字、情绪等。用简洁的中文回答。"
        if is_sticker:
            prompt = "这是一张动画表情包。请描述它的画面内容、文字(如果有)以及表达的情绪。用简洁的中文回答。"
        
        for attempt in range(retry):
            try:
                logger.debug(f"[Vision] 尝试 {attempt + 1}/{retry}")
                logger.debug(f"[Vision] 调用参数: model={self.model}, max_tokens=300")
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    }],
                    max_tokens=300
                )
                
                result = response.choices[0].message.content or "[图片识别无结果]"
                
                if is_sticker:
                    result = f"[动画表情] {result}"
                
                logger.debug(f"[Vision] 识别成功: {result[:100]}...")
                logger.debug(f"[Vision] Token 使用: prompt={response.usage.prompt_tokens if response.usage else 'N/A'}, completion={response.usage.completion_tokens if response.usage else 'N/A'}")
                return result
            
            except Exception as e:
                logger.error(f"[Vision] 识别失败 (尝试 {attempt + 1}/{retry}): {type(e).__name__}: {str(e)}")
                if attempt == retry - 1:
                    return f"[图片识别失败: {str(e)}]"
                await asyncio.sleep(2 ** attempt)
        
        return "[图片识别失败]"

class ChatClient:
    """聊天模型客户端 - 专注理解与生成"""
    
    def __init__(self):
        config = ConfigManager()
        self.api_key = config.get_env("chat_api_key")
        self.base_url = config.get_env("chat_api_url")
        self.model = config.get_env("chat_model_name", "deepseek-chat")
        
        if not self.api_key or not self.base_url:
            logger.error("Chat Client 配置不完整，请检查 .env.dev 文件中的 chat_api_key 和 chat_api_url")
        
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
        disable_thinking: bool = False,
        json_mode: bool = False,
    ) -> Optional[str]:
        """通用对话生成，支持参数化配置"""

        kwargs: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if disable_thinking:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        for attempt in range(retry):
            try:
                logger.debug(f"[Chat] 尝试 {attempt + 1}/{retry} (temperature={temperature}, thinking={'off' if disable_thinking else 'on'}, json={json_mode})")
                
                response = await self.client.chat.completions.create(**kwargs)
                
                message = response.choices[0].message
                reasoning = getattr(message, 'reasoning_content', None)
                if reasoning:
                    logger.debug(f"[Chat] 思考过程 ({len(reasoning)} 字): {reasoning[:100]}...")
                
                result = message.content or "..."
                logger.success(f"[Chat] 生成成功: {result[:100]}...")
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
        self, messages: Sequence[ChatCompletionMessageParam], retry: int = 3
    ) -> Optional[str]:
        """生成聊天回复 - thinking=disabled 保持低延迟"""
        return await self.generate_response(
            messages, retry, temperature=0.8, disable_thinking=True
        )

    async def generate_summary(self, context: str, retry: int = 3) -> Optional[str]:
        """生成 L1 摘要 - 格式提示词放在 system 第一条"""
        FORMAT_PROMPT = """[输出格式硬性要求 - 最高优先级]
请严格按以下格式输出，每条占一行：
1. [时间/时间段] [昵称] 做了什么/说了什么，简要内容。
2. [时间/时间段] [昵称] 做了什么/说了什么，简要内容。
...
禁止输出任何额外说明、Markdown 标记或代码块。
""".strip()

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": FORMAT_PROMPT},
            {"role": "user", "content": (
                "总结以下群聊对话，忽略寒暄和无关细节，保留主要事件和讨论点。"
                "每条总结务必包含：1) 触发时间（可用消息时间或大致时间段）；"
                "2) 相关发送人昵称（ context 中的前缀已经包含昵称，请沿用）；"
                "3) 简要内容。"
                "以列表形式输出，确保可追溯到是谁在什么时候做了什么。\n\n"
                f"{context}"
            )},
        ]

        result = await self.generate_response(
            messages, retry, temperature=0.5, disable_thinking=True
        )
        if not result:
            logger.error("[Chat] 生成总结失败，返回 None")
            return None

        # 格式校验：至少要有数字编号行
        if not any(line.strip()[:1].isdigit() for line in result.split("\n") if line.strip()):
            logger.warning("[Summary] 格式异常，尝试修复重试")
            fix_msg: ChatCompletionMessageParam = {
                "role": "user",
                "content": (
                    "你的输出格式不正确。请重新输出，严格按编号列表格式，每条以数字开头。"
                    "禁止任何额外文字。\n\n原始对话:\n" + context
                ),
            }
            result = await self.generate_response(
                [messages[0], fix_msg], 1, temperature=0.3, disable_thinking=True
            )

        return result

    async def extract_facts_v2(self, context: str, retry: int = 3) -> List[Dict[str, Any]]:
        """
        提取重要事实，使用 reasoning + JSON 结构输出。
        格式提示词放在 system 第一条。
        """
        FORMAT_PROMPT = """[输出格式硬性要求 - 最高优先级]
你必须且只能输出一个严格的 JSON 数组。禁止任何其他输出。
每个元素的结构：
{"type": "plan|attribute|commitment", "person": "昵称", "content": "用第三人称叙述的事实", "confidence": "high|medium"}

没有值得记录的信息时输出：[]

正确示例：
[
  {"type": "plan", "person": "小明", "content": "小明下周三去上海出差，周四返回", "confidence": "high"},
  {"type": "attribute", "person": "小红", "content": "小红生日是 5 月 20 日", "confidence": "high"}
]
""".strip()

        TASK_PROMPT = """分析以下群聊对话，提取值得永久记忆的信息。

[判断标准 - 仅记录以下三类]
1. plan（计划）：包含具体时间/地点的约定
2. attribute（属性）：生日、职业、过敏原、住址城市、专业领域等
3. commitment（承诺）：认真表达的承诺（排除"改天请你吃饭"类客套）

[绝对排除]
日常闲聊、情绪表达（"今天好累"）、模糊意图（"我想学日语"——没有计划时间不算）、对当下事件的评论（"这游戏真好玩"）、任何一个月内失去意义的内容。

对话记录：
""" + context

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": FORMAT_PROMPT},
            {"role": "user", "content": TASK_PROMPT},
        ]

        raw = await self.generate_response(
            messages, retry, temperature=0.3
            # 不传 disable_thinking，启用推理
        )
        if not raw:
            return []

        # 格式校验 + 重试
        for attempt in range(2):
            try:
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                    cleaned = re.sub(r"\s*```$", "", cleaned)
                data = json.loads(cleaned)
                if isinstance(data, list):
                    return [item for item in data if isinstance(item, dict)]
                if isinstance(data, dict):
                    inner = data.get("facts") or data.get("results") or data.get("items")
                    if isinstance(inner, list):
                        return inner
                logger.warning(f"[Facts] 非预期结构: {type(data)}")
            except Exception as e:
                logger.warning(f"[Facts] JSON 解析失败 (attempt {attempt+1}): {e}")

            if attempt == 0:
                fix_msg: ChatCompletionMessageParam = {
                    "role": "user",
                    "content": (
                        "你的输出 JSON 格式不正确。请严格按 system prompt 中定义的格式重新输出，"
                        "只输出 JSON 数组，不要带任何其他内容。"
                    ),
                }
                raw = await self.generate_response(
                    [messages[0], fix_msg], 1, temperature=0.2, disable_thinking=True
                )
                if not raw:
                    return []

        logger.error("[Facts] 经过修复仍无法解析，返回空列表")
        return []

    async def extract_maintenance_actions(
        self, prompt: str, retry: int = 2
    ) -> Optional[Dict[str, Any]]:
        """
        AI 审核事实库：标记过期、检测矛盾、建议合并。
        格式提示词放在 system 第一条。
        """
        FORMAT_PROMPT = """[输出格式硬性要求 - 最高优先级]
你必须且只能输出一个严格的 JSON 对象，格式：
{
  "expire": [],
  "conflicts": [{"obsolete": 0, "keep": 0, "reason": "冲突原因"}],
  "merges": [{"merge_ids": [], "merged_content": "合并后的完整内容"}]
}
禁止任何额外文字、Markdown 标记或代码块。
""".strip()

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": FORMAT_PROMPT},
            {"role": "user", "content": prompt},
        ]

        raw = await self.generate_response(
            messages, retry, temperature=0.3, json_mode=True
        )
        if not raw:
            return None

        for attempt in range(2):
            try:
                cleaned = raw.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
                    cleaned = re.sub(r"\s*```$", "", cleaned)
                data = json.loads(cleaned)
                if isinstance(data, dict):
                    return data
            except Exception as e:
                logger.warning(f"[Maintenance] JSON 解析失败 (attempt {attempt+1}): {e}")

            if attempt == 0:
                fix_msg: ChatCompletionMessageParam = {
                    "role": "user",
                    "content": "你的 JSON 格式不正确，请仅重新输出正确的 JSON 对象，禁止额外内容。",
                }
                raw = await self.generate_response(
                    [messages[0], fix_msg], 1, temperature=0.2, disable_thinking=True
                )
                if not raw:
                    return None

        return None

    async def generate_daily_summary_data(self, context: str, retry: int = 3) -> str:
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

群聊记录：
{context}"""

        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": FORMAT_PROMPT},
            {"role": "user", "content": TASK_PROMPT},
        ]

        # 尝试使用 JSON 模式 (如果模型支持)
        try:
            raw = await self.generate_response(
                messages, retry, temperature=0.8, json_mode=True
            )
            if raw:
                return raw
        except Exception as e:
            logger.debug(f"[DailySummary] JSON 模式不可用，回退普通模式: {e}")

        # 回退到普通模式
        fallback = await self.generate_response(
            messages, retry, temperature=0.8, disable_thinking=True
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
            logger.warning("Embedding Client 配置不完整，知识库功能可能受限。请检查 .env.dev 文件中的 embedding_api_key 和 embedding_api_url")
        
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
