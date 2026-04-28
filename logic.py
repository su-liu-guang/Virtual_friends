import hashlib
import aiohttp
import re
from datetime import datetime
from typing import List, Optional, Callable
from nonebot import logger
from openai.types.chat import ChatCompletionMessageParam
from .database import Message, ImageCache, Summary, ImportantEvent
from .config import ConfigManager

ERROR_CAPTION_PREFIXES = ("[图片识别失败", "[Vision Error")

# ====== 格式约束提示词 — 放在 system 最开头 ======
OUTPUT_FORMAT_PROMPT = """[输出格式硬性要求 - 最高优先级]
你必须且只能输出以下格式：
<persona_reply>你的回复内容</persona_reply>

[硬性规则]
1. 必须且只能输出一组完整的 <persona_reply> 和 </persona_reply>
2. 标签外绝对不能有任何文字、说明、旁白、Markdown 标记
3. 你的回复必须严格以 <persona_reply> 开头，以 </persona_reply> 结尾
4. 禁止输出多组标签块

正确示例：<persona_reply>早上好呀，今天天气真不错呢！</persona_reply>
""".strip()

# ====== 格式修正重试 system prompt — 放在 retry context 最开头 ======
FORMAT_RETRY_SYSTEM = """[格式修正 - 最高优先级]
你之前的回复格式不正确。现在你必须且只能输出一组完整的标签：
<persona_reply>你的回复内容</persona_reply>
禁止任何标签外的文字、解释或 Markdown 符号。
""".strip()


def has_complete_persona_reply_tag(raw_text: Optional[str]) -> bool:
    """检查是否包含完整的 persona_reply 标签对。"""
    if not raw_text:
        return False
    text = raw_text.strip()
    if not text:
        return False
    return re.search(r"<persona_reply>\s*.*?\s*</persona_reply>", text, flags=re.IGNORECASE | re.DOTALL) is not None


def sanitize_persona_reply(raw_text: Optional[str]) -> str:
    """清洗模型输出，仅保留人设台词内容。"""
    if not raw_text:
        return ""

    text = raw_text.strip()
    if not text:
        return ""

    # 优先提取强约束标签中的内容
    tagged = re.search(r"<persona_reply>\s*(.*?)\s*</persona_reply>", text, flags=re.IGNORECASE | re.DOTALL)
    if tagged:
        text = tagged.group(1).strip()
    else:
        # 半闭合容错: 只有开标签或只有闭标签时，尽量提取正文
        open_only = re.search(r"<persona_reply>\s*(.*)$", text, flags=re.IGNORECASE | re.DOTALL)
        close_only = re.search(r"^(.*?)\s*</persona_reply>", text, flags=re.IGNORECASE | re.DOTALL)
        if open_only:
            text = open_only.group(1).strip()
        elif close_only:
            text = close_only.group(1).strip()

    # 去掉常见 markdown 包裹
    text = re.sub(r"^```(?:[a-zA-Z]+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # 去除可能的时间戳前缀 [2025-12-12 19:30]
    text = re.sub(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]\s*", "", text)

    # 逐行移除常见元信息和说话人前缀
    cleaned_lines: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if re.match(r"^\{\{发送图片:.*\}\}$", line):
            cleaned_lines.append(line)
            continue

        if re.match(r"^(?:说明|注[:：]|系统|旁白|内心|思考|analysis|assistant|user)\b", line, flags=re.IGNORECASE):
            continue

        line = re.sub(
            r"^(?:\[[^\]]{1,20}\]|（[^）]{1,20}）|\([^)]{1,20}\)|[A-Za-z\u4e00-\u9fa5]{1,20})\s*[：:]\s*",
            "",
            line,
        )
        if line:
            cleaned_lines.append(line)

    return "\n".join(cleaned_lines).strip()


def is_valid_caption(caption: Optional[str]) -> bool:
    if not caption:
        return False
    stripped = caption.strip()
    if not stripped:
        return False
    return not stripped.startswith(ERROR_CAPTION_PREFIXES)


async def process_image_message(image_url: str, vision_client, caption_override: Optional[str] = None, is_sticker: bool = False) -> str:
    """处理图片消息,实现缓存优先策略"""
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as resp:
            image_data = await resp.read()
    
    md5 = hashlib.md5(image_data).hexdigest()
    
    cached = await ImageCache.get_or_none(md5=md5)
    if cached:
        return md5
    
    if caption_override:
        caption = caption_override
    else:
        caption = await vision_client.recognize_image(image_url, is_sticker=is_sticker)
    
    if is_valid_caption(caption):
        await ImageCache.create(md5=md5, caption=caption)
    else:
        logger.warning("[Vision] 识别结果无效，已跳过缓存写入（请检查模型是否支持图像识别）")
    
    return md5


async def generate_with_format_retry(
    chat_client,
    messages: List[ChatCompletionMessageParam],
    validator: Callable[[str], bool],
    retry_prompt_system: str,
    max_retries: int = 1,
) -> Optional[str]:
    """
    通用格式校验 + 重试函数。
    
    发送消息 → 校验格式 → 失败则将格式提示词作为 system 第一条重试。
    retry 时 FORMAT_PROMPT 放在 context 最前面（system），原有消息保留在后。
    """
    raw = await chat_client.generate_chat_reply(messages, retry=3)
    if not raw:
        return None

    if validator(raw):
        return raw

    # 格式校验失败，构造修正型重试
    logger.warning("[Format] 首次输出格式不完整，构造修正重试")
    retry_messages: List[ChatCompletionMessageParam] = [
        {"role": "system", "content": retry_prompt_system},
        *messages,
    ]
    repaired = await chat_client.generate_chat_reply(retry_messages, retry=1)
    if repaired and validator(repaired):
        return repaired

    return raw  # 回退，交给 sanitize 清洗


class ContextBuilder:
    """上下文构建器"""
    
    def __init__(self, config_manager: ConfigManager, knowledge_base=None):
        self.config_manager = config_manager
        self.knowledge_base = knowledge_base
    
    async def build_context(
        self, 
        group_id: str, 
        user_nickname: str, 
        current_time: datetime
    ) -> List[ChatCompletionMessageParam]:
        """构建完整的对话上下文"""
        config = self.config_manager.get_instance_config(group_id)
        persona_prompt = self.config_manager.get_persona_prompt(config["persona_name"])
        
        # Recent Layer - 最近 50 条消息（无论是否已处理）
        recent_msgs_desc = (
            await Message.filter(group_id=group_id)
            .order_by("-timestamp", "-id")
            .limit(50)
            .all()
        )
        recent_msgs = list(reversed(recent_msgs_desc))

        # ---- L1/L2/L3 全量摘要 ----
        all_l1 = (
            await Summary.filter(group_id=group_id, level=1, is_archived=False)
            .order_by("-created_at")
            .all()
        )
        all_l2 = (
            await Summary.filter(group_id=group_id, level=2, is_archived=False)
            .order_by("-created_at")
            .all()
        )
        all_l3 = (
            await Summary.filter(group_id=group_id, level=3, is_archived=False)
            .order_by("-created_at")
            .all()
        )

        # ---- ImportantEvent 直连查询 ----
        all_facts = await ImportantEvent.filter(
            group_id=group_id, validity=True
        ).order_by("-recorded_date").all()

        # ---- Knowledge Layer ----
        knowledge_text = ""
        last_user_msg = None
        if recent_msgs:
            for msg in reversed(recent_msgs):
                if msg.role == "user":
                    last_user_msg = msg
                    break

        if self.knowledge_base and last_user_msg and last_user_msg.content and len(last_user_msg.content) > 2:
            chunks = await self.knowledge_base.search(last_user_msg.content, top_k=1)
            if chunks:
                knowledge_text = "\n[参考资料]:\n"
                for i, chunk in enumerate(chunks):
                    knowledge_text += f"{i+1}. (来自 {chunk['source']} - {chunk['title']})\n{chunk['text']}\n"
                
                knowledge_text += "\n[图片发送规则]\n参考资料中可能会提到图片资源，格式为 `(此处有一张图片，名称为：xxx)`。\n如果你认为展示该图片有助于回答用户问题，请在回复的末尾单独一行输出：`{{发送图片:xxx}}`。\n请只发送与问题高度相关的图片。\n"

        # ---- 时间信息 ----
        weekday_cn = ["一", "二", "三", "四", "五", "六", "日"]
        weekday = weekday_cn[current_time.isoweekday() - 1]
        time_str = f"{current_time.strftime('%Y年%m月%d日')} 星期{weekday} {current_time.strftime('%H:%M')}"
        
        # ====== System Content — 格式要求放在最前面 ======
        system_content = OUTPUT_FORMAT_PROMPT  # 第一行就是格式约束
        
        system_content += f"\n\n{persona_prompt}"
        system_content += f"\n\n[当前时间]: {time_str}"
        system_content += f"\n[用户称呼]: {user_nickname}"
        system_content += f"\n{knowledge_text}"

        # ---- 全量摘要注入 ----
        if all_l3:
            system_content += "\n\n[宏观印象 L3]:\n"
            for s3 in all_l3:
                system_content += f"- ({s3.time_range}) {s3.content.strip()}\n"

        if all_l2:
            system_content += "\n\n[叙事概括 L2]:\n"
            for s2 in all_l2:
                system_content += f"- ({s2.time_range}) {s2.content.strip()}\n"

        if all_l1:
            system_content += "\n\n[近期详情 L1]:\n"
            for s1 in all_l1:
                system_content += f"- ({s1.time_range}) {s1.content.strip()}\n"

        # ---- 事实注入（近期 vs 远期分层）----
        if all_facts:
            now_date = current_time.date()
            recent_facts = [f for f in all_facts if (now_date - f.recorded_date).days <= 14]
            old_facts = [f for f in all_facts if (now_date - f.recorded_date).days > 14]
            if recent_facts:
                system_content += "\n\n[近期重要事实]:\n"
                for f in recent_facts:
                    prefix = "[高] " if f.confidence == "high" else ""
                    system_content += f"- {prefix}{f.event_content}\n"
            if old_facts:
                system_content += "\n\n[更早的事实]:\n"
                for f in old_facts[:10]:
                    system_content += f"- {f.event_content}\n"
        
        messages: List[ChatCompletionMessageParam] = [{"role": "system", "content": system_content.strip()}]
        
        # ---- Recent Layer ----
        for msg in recent_msgs:
            content = msg.content
            
            if msg.image_md5:
                cache = await ImageCache.get_or_none(md5=msg.image_md5)
                if cache:
                    content = f"[系统注解: 图片内容为 {cache.caption}]\n{content}" if content else f"[系统注解: 图片内容为 {cache.caption}]"
            
            local_ts = msg.timestamp.astimezone()
            time_prefix = f"[{local_ts.strftime('%Y-%m-%d %H:%M')}] "
            
            if msg.role == "user" and msg.user_nickname:
                content = f"{time_prefix}{msg.user_nickname}: {content}" if content else f"{time_prefix}{msg.user_nickname}: [无文本内容]"
            elif msg.role == "ai":
                content = f"{time_prefix}{content}"

            role = "assistant" if msg.role == "ai" else "user"
            message: ChatCompletionMessageParam = {
                "role": role,  # type: ignore
                "content": content
            }
            messages.append(message)

        return messages
