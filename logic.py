import hashlib
import aiohttp
import re
from datetime import datetime
from typing import List, Optional
from nonebot import logger
from openai.types.chat import ChatCompletionMessageParam
from .database import Message, ImageCache, Summary
from .config import ConfigManager

ERROR_CAPTION_PREFIXES = ("[图片识别失败", "[Vision Error")

OUTPUT_FORMAT_TAIL_PROMPT = """
[输出格式要求]
你只输出人设正在说的话，不要输出任何额外说明、角色分析、旁白、动作描写或思考过程。
你必须且只能输出一个完整标签块，严格使用以下格式：
<persona_reply>你的回复内容</persona_reply>

[硬性规则]
1. 必须同时包含开始标签 <persona_reply> 和结束标签 </persona_reply>。
2. 禁止输出第二个 persona_reply 标签块。
3. 禁止输出 markdown 代码块、解释文字、前后缀说明。
4. 在输出前先自检：若标签不闭合，必须先补全，再输出。

[兜底模板]
如果你不确定如何回答，也必须输出：
<persona_reply>...</persona_reply>
""".strip()

PERSONA_REPLY_RETRY_PROMPT = (
    "你上次输出的 <persona_reply> 标签格式不完整或不合规。"
    "现在只允许返回一个完整标签块，且必须闭合："
    "<persona_reply>你的回复内容</persona_reply>。"
    "禁止返回任何额外文字、解释、代码块或多标签。"
)


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

        # 例: 小明: 你好 / [小明]: 你好 / （小明）：你好
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
    # 下载图片计算 MD5
    async with aiohttp.ClientSession() as session:
        async with session.get(image_url) as resp:
            image_data = await resp.read()
    
    md5 = hashlib.md5(image_data).hexdigest()
    
    # 查询缓存
    cached = await ImageCache.get_or_none(md5=md5)
    if cached:
        return md5
    
    # 如果有强制指定的描述（如表情包），则跳过 Vision API
    if caption_override:
        caption = caption_override
    else:
        # 调用 Vision API
        caption = await vision_client.recognize_image(image_url, is_sticker=is_sticker)
    
    # 写入缓存（仅在识别结果有效时）
    if is_valid_caption(caption):
        await ImageCache.create(md5=md5, caption=caption)
    else:
        logger.warning("[Vision] 识别结果无效，已跳过缓存写入（请检查模型是否支持图像识别）")
    
    return md5

class ContextBuilder:
    """上下文构建器"""
    
    def __init__(self, config_manager: ConfigManager, knowledge_base=None, memory_retriever=None):
        self.config_manager = config_manager
        self.knowledge_base = knowledge_base
        self.memory_retriever = memory_retriever
    
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
        # 还原为时间正序，便于模型理解对话流
        recent_msgs = list(reversed(recent_msgs_desc))

        # L1 Layer - 最近一条 L1 摘要
        latest_l1_summary = (
            await Summary.filter(group_id=group_id, level=1, is_archived=False)
            .order_by("-created_at", "-id")
            .first()
        )

        # Knowledge & Memory Layer
        knowledge_text = ""
        memory_hits = []
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

        if self.memory_retriever and last_user_msg and last_user_msg.content and len(last_user_msg.content) > 2:
            memory_hits = await self.memory_retriever.retrieve(
                group_id,
                last_user_msg.content,
                top_k=12,
                final_k=5
            )

        # 时间信息
        weekday_cn = ["一", "二", "三", "四", "五", "六", "日"]
        weekday = weekday_cn[current_time.isoweekday() - 1]
        time_str = f"{current_time.strftime('%Y年%m月%d日')} 星期{weekday} {current_time.strftime('%H:%M')}"
        
        # System Layer
        system_content = f"""{persona_prompt}

[当前时间]: {time_str}
[用户称呼]: {user_nickname}
{knowledge_text}
"""

        # 默认尾部提示词：约束模型仅输出人设台词，便于后续正则清洗
        system_content += f"\n\n{OUTPUT_FORMAT_TAIL_PROMPT}"
        
        # Memory Layer - 使用检索结果，减少 token
        if memory_hits:
            system_content += "\n[记忆回顾]:\n"
            for item in memory_hits:
                system_content += f"- ({item['tag']}) {item['content']}\n"

        if latest_l1_summary and latest_l1_summary.content:
            system_content += (
                "\n[最近L1总结]:\n"
                f"时间范围: {latest_l1_summary.time_range}\n"
                f"{latest_l1_summary.content.strip()}\n"
            )
        
        messages: List[ChatCompletionMessageParam] = [{"role": "system", "content": system_content.strip()}]
        
        # Recent Layer - 消息已在上面获取，这里直接使用
        for msg in recent_msgs:
            content = msg.content
            
            # 图片处理
            if msg.image_md5:
                cache = await ImageCache.get_or_none(md5=msg.image_md5)
                if cache:
                    content = f"[系统注解: 图片内容为 {cache.caption}]\n{content}" if content else f"[系统注解: 图片内容为 {cache.caption}]"
            
            # 添加时间戳 (本地时间)
            local_ts = msg.timestamp.astimezone()
            time_prefix = f"[{local_ts.strftime('%Y-%m-%d %H:%M')}] "
            
            if msg.role == "user" and msg.user_nickname:
                content = f"{time_prefix}{msg.user_nickname}: {content}" if content else f"{time_prefix}{msg.user_nickname}: [无文本内容]"
            elif msg.role == "ai":
                # AI 的消息也加上时间，保持格式一致
                content = f"{time_prefix}{content}"

            role = "assistant" if msg.role == "ai" else "user"
            message: ChatCompletionMessageParam = {
                "role": role,  # type: ignore
                "content": content
            }
            messages.append(message)
        
        return messages
