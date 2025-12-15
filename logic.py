import hashlib
import aiohttp
from datetime import datetime
from typing import List, Optional
from nonebot import logger
from openai.types.chat import ChatCompletionMessageParam
from .database import Message, ImageCache, Summary, ImportantEvent
from .config import ConfigManager

ERROR_CAPTION_PREFIXES = ("[图片识别失败", "[Vision Error")


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
        
        # Recent Layer - 最近未总结的消息（is_processed=False）
        # 提前获取以便用于检索
        recent_msgs = (
            await Message.filter(group_id=group_id, is_processed=False)
            .order_by("timestamp", "id")
            .limit(100)
            .all()
        )

        # Knowledge Layer - 知识库检索
        knowledge_text = ""
        if self.knowledge_base and recent_msgs:
            # 找到最后一条用户消息
            last_user_msg = None
            for msg in reversed(recent_msgs):
                if msg.role == "user":
                    last_user_msg = msg
                    break
            
            if last_user_msg and last_user_msg.content and len(last_user_msg.content) > 2:
                chunks = await self.knowledge_base.search(last_user_msg.content, top_k=3)
                if chunks:
                    knowledge_text = "\n[参考资料]:\n"
                    for i, chunk in enumerate(chunks):
                        knowledge_text += f"{i+1}. (来自 {chunk['source']} - {chunk['title']})\n{chunk['text']}\n"
                    
                    knowledge_text += "\n[图片发送规则]\n参考资料中可能会提到图片资源，格式为 `(此处有一张图片，名称为：xxx)`。\n如果你认为展示该图片有助于回答用户问题，请在回复的末尾单独一行输出：`{{发送图片:xxx}}`。\n请只发送与问题高度相关的图片。\n"

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
        
        # Fact Layer - 长期记忆
        facts = await ImportantEvent.filter(group_id=group_id, validity=True).all()
        if facts:
            facts_text = "\n".join([f"- {f.event_content}" for f in facts])
            system_content += f"\n[已知事实]:\n{facts_text}\n"
        
        # Memory Layer - 流式记忆
        l3_summaries = await Summary.filter(group_id=group_id, level=3).all()
        l2_summaries = await Summary.filter(group_id=group_id, level=2, is_archived=False).all()
        l1_summaries = await Summary.filter(group_id=group_id, level=1, is_archived=False).all()
        
        if l3_summaries or l2_summaries or l1_summaries:
            system_content += "\n[记忆回顾]:\n"
            
            for s in l3_summaries:
                system_content += f"[宏观印象 {s.time_range}]: {s.content}\n"
            for s in l2_summaries:
                system_content += f"[叙事概括 {s.time_range}]: {s.content}\n"
            for s in l1_summaries:
                system_content += f"[细节摘要 {s.time_range}]: {s.content}\n"
        
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
