import hashlib
import aiohttp
import re
from datetime import datetime
from typing import List, Optional, Callable, Tuple
from nonebot import logger
from openai.types.chat import ChatCompletionMessageParam
from .database import Message, ImageCache, Summary
from .config import ConfigManager

ERROR_CAPTION_PREFIXES = ("[图片识别失败", "[Vision Error")

# ====== 格式约束提示词 — 放在 system 最开头 ======
OUTPUT_FORMAT_PROMPT = """[输出格式硬性约束 - 必须严格遵守]

你的完整回复必须精确等于以下格式，一个多余字符都不允许：

<persona_reply>你的回复内容</persona_reply>

规则：
1. 回复的第一个字符必须是 <persona_reply> 的 <
2. 回复的最后一个字符必须是 </persona_reply> 的 >
3. 标签外不允许存在任何内容：无换行、无空格、无解释文字、无 Markdown
4. 禁止在标签前添加时间戳、昵称、思考过程、旁白、引导语

错误示例（这些都是被禁止的）：
  [2025-01-01 12:00] <persona_reply>你好</persona_reply>
  <persona_reply>你好</persona_reply>（希望你喜欢）
  ````<persona_reply>你好</persona_reply>````
  好的，我来回复：<persona_reply>你好</persona_reply>

正确示例（这是唯一允许的格式）：
<persona_reply>你好呀，今天天气真好！</persona_reply>
""".strip()

# ====== 放在 system prompt 末尾的格式再次提醒（利用近因效应）======
FORMAT_REMINDER = "\n\n[再次强调] 你的回复必须等于 <persona_reply>回复内容</persona_reply>，标签外零字符。"

# ====== 放在最后一条 user 消息末尾的短格式提醒（利用近因效应）======
SHORT_FORMAT_REMINDER = "\n\n[输出要求] 只输出 <persona_reply>回复内容</persona_reply>"

# ====== 格式修正重试 system prompt — 放在 retry context 最开头 ======
FORMAT_RETRY_SYSTEM = """[格式修正 - 最高优先级]
你上一次的回复格式不正确。请重新回复，严格输出：
<persona_reply>你的回复内容</persona_reply>
零额外字符，零标签外内容。""".strip()

# ====== 知识库命中时注入的精确输出指令（优先级高于人设）======
KNOWLEDGE_PRECISION_INSTRUCTION = """[资料库精确输出指令 - 优先级高于人设]

检测到知识库中存在与该问题相关的参考资料，请严格遵守以下规则：
1. 将参考资料中所有相关信息**完整、逐条**输出，不得省略关键细节
2. 严格基于参考资料原文，**禁止编造或补充**参考资料中不存在的信息
3. 如果资料包含具体数据、名称、步骤，必须全部包含在回复中
4. 上述精确输出要求**高于人设聊天风格**——资料完整性优先于角色扮演
5. 如果资料不足，明确说明，不要编造""".strip()

def _build_user_message_instructions() -> str:
    """构建追加到首个 user 消息末尾的纯格式+沉浸指令。
    
    这些指令是纯静态文本，不影响 LLM 缓存命中率。
    动态上下文（时间/用户/knowledge/L1摘要）已移到当前 user 消息前缀中。
    """
    return """

[输出格式要求] 你的回复必须严格以 <persona_reply> 开头、以 </persona_reply> 结尾，标签外零字符。

【角色沉浸要求】在你的思考过程（<think>标签内）中，请遵守以下规则：
1. 请以角色第一人称进行内心独白，用括号包裹内心活动，例如"（心想：……）"或"(内心OS：……)"
2. 用第一人称描写角色的内心感受，例如"我心想""我觉得""我暗自"等
3. 思考内容应沉浸在角色中，通过内心独白分析剧情和规划回复"""


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

    result = "\n".join(cleaned_lines).strip()
    result = re.sub(r"（[^）]*）", "", result)
    result = re.sub(r"\([^)]*\)", "", result)
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result


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
) -> Tuple[Optional[str], Optional[str]]:
    """
    通用格式校验 + 重试函数。

    发送消息 → 校验格式 → 失败则将格式提示词合并到已有 system 消息中重试。
    返回 (raw_content, reasoning_content) 元组。
    """
    result = await chat_client.generate_chat_reply(messages, retry=3)
    if not result:
        return (None, None)

    raw, reasoning = result

    if validator(raw):
        return (raw, reasoning)

    # 格式校验失败，将修正指令合并到已有 system 消息中重试
    logger.warning("[Format] 首次输出格式不完整，构造修正重试")
    retry_messages: List[ChatCompletionMessageParam] = []
    system_found = False
    for msg in messages:
        if msg["role"] == "system" and not system_found:
            system_found = True
            retry_messages.append({
                "role": "system",
                "content": retry_prompt_system + "\n\n---\n\n" + msg["content"]  # type: ignore[dict-item]
            })
        else:
            retry_messages.append(msg)
    repaired_result = await chat_client.generate_chat_reply(retry_messages, retry=1)
    if repaired_result:
        repaired_raw, repaired_reasoning = repaired_result
        if validator(repaired_raw):
            return (repaired_raw, repaired_reasoning)

    return (raw, reasoning)  # 回退，交给 sanitize 清洗


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
        
        # Recent Layer - 最近 25 条消息（无论是否已处理）
        recent_msgs_desc = (
            await Message.filter(group_id=group_id)
            .order_by("-timestamp", "-id")
            .limit(25)
            .all()
        )
        recent_msgs = list(reversed(recent_msgs_desc))

        # ---- L1/L2/L3 摘要（加 limit 截断） ----
        all_l1 = (
            await Summary.filter(group_id=group_id, level=1, is_archived=False)
            .order_by("created_at")
            .limit(15)
            .all()
        )
        all_l2 = (
            await Summary.filter(group_id=group_id, level=2, is_archived=False)
            .order_by("created_at")
            .limit(10)
            .all()
        )
        all_l3 = (
            await Summary.filter(group_id=group_id, level=3, is_archived=False)
            .order_by("created_at")
            .limit(5)
            .all()
        )

        # ---- Knowledge Layer ----
        knowledge_text = ""
        last_user_msg = None
        if recent_msgs:
            for msg in reversed(recent_msgs):
                if msg.role == "user":
                    last_user_msg = msg
                    break

        if self.knowledge_base and last_user_msg and last_user_msg.content and len(last_user_msg.content) > 2:
            knowledge_query = last_user_msg.content
            chunks = await self.knowledge_base.search(knowledge_query, top_k=5)
            if chunks:
                knowledge_text = "\n[参考资料]:\n"
                for i, chunk in enumerate(chunks):
                    knowledge_text += f"{i+1}. (来自 {chunk['source']} - {chunk['title']})\n{chunk['text']}\n"
                
                knowledge_text += "\n[图片发送规则]\n参考资料中可能会提到图片资源，格式为 `(此处有一张图片，名称为：xxx)`。\n如果你认为展示该图片有助于回答用户问题，请在回复的末尾单独一行输出：`{{发送图片:xxx}}`。\n请只发送与问题高度相关的图片。\n"

        # ---- 时间信息 ----
        weekday_cn = ["一", "二", "三", "四", "五", "六", "日"]
        weekday = weekday_cn[current_time.isoweekday() - 1]
        time_str = f"{current_time.strftime('%Y年%m月%d日')} 星期{weekday} {current_time.strftime('%H:%M')}"
        
        # ====== SYSTEM MESSAGE: 静态 + 准静态（最大化 LLM 缓存命中）======
        system_content = OUTPUT_FORMAT_PROMPT
        system_content += f"\n\n{persona_prompt}"

        # L3 宏观印象 — 数月不变，准静态
        if all_l3:
            system_content += "\n\n[宏观印象 L3]:\n"
            for s3 in all_l3:
                system_content += f"- ({s3.time_range}) {s3.content.strip()}\n"

        # L2 叙事概括 — 数天不变，相对稳定
        if all_l2:
            system_content += "\n\n[叙事概括 L2]:\n"
            for s2 in all_l2:
                system_content += f"- ({s2.time_range}) {s2.content.strip()}\n"

        # L1 近期详情 — 一天内基本不变，准静态
        if all_l1:
            system_content += "\n\n[近期详情 L1]:\n"
            for s1 in all_l1:
                system_content += f"- ({s1.time_range}) {s1.content.strip()}\n"

        system_content += FORMAT_REMINDER

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
            if msg.role == "ai" and msg.reasoning_content:
                message["reasoning_content"] = msg.reasoning_content  # type: ignore[index]
            messages.append(message)

        # ====== 第一条 user 不再注入动态知识库，避免污染后续历史消息缓存 ======
        context_prefix = ""
        for m in messages:
            if m["role"] == "user":
                cur = m.get("content", "")
                if isinstance(cur, str):
                    m["content"] = context_prefix + cur  # type: ignore[index]
                break

        # ====== 参考资料 + 时间前缀：注入到最后一条 user 消息（紧邻当前问题）======
        time_prefix = f"[当前时间]: {time_str}\n[当前对话对象]: {user_nickname}\n\n"
        last_user_prefix = time_prefix
        if knowledge_text.strip():
            last_user_prefix = knowledge_text.strip() + "\n\n" + KNOWLEDGE_PRECISION_INSTRUCTION + "\n\n" + time_prefix

        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                cur = messages[i].get("content", "")
                if isinstance(cur, str):
                    messages[i]["content"] = last_user_prefix + cur + SHORT_FORMAT_REMINDER  # type: ignore[index]
                break

        return messages
