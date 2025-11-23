import hashlib
import aiohttp
from datetime import datetime
from typing import List, Optional
from openai.types.chat import ChatCompletionMessageParam
from .database import Message, ImageCache, Summary, ImportantEvent
from .config import ConfigManager

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
    
    # 写入缓存
    await ImageCache.create(md5=md5, caption=caption)
    
    return md5

class ContextBuilder:
    """上下文构建器"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config_manager = config_manager
    
    async def build_context(
        self, 
        group_id: str, 
        user_nickname: str, 
        current_time: datetime
    ) -> List[ChatCompletionMessageParam]:
        """构建完整的对话上下文"""
        config = self.config_manager.get_instance_config(group_id)
        persona_prompt = self.config_manager.get_persona_prompt(config["persona_name"])
        
        # 时间信息
        weekday_cn = ["一", "二", "三", "四", "五", "六", "日"]
        weekday = weekday_cn[current_time.isoweekday() - 1]
        time_str = f"{current_time.strftime('%Y年%m月%d日')} 星期{weekday} {current_time.strftime('%H:%M')}"
        
        # System Layer
        system_content = f"""{persona_prompt}

[当前时间]: {time_str}
[用户称呼]: {user_nickname}
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
        
        # Recent Layer - 最近对话
        # 使用时间+ID双重排序，确保消息顺序绝对正确（防止系统时间回拨等极端情况）
        recent_msgs = await Message.filter(group_id=group_id).order_by("timestamp", "id").limit(100).all()
        
        for msg in recent_msgs:
            content = msg.content
            
            # 图片处理
            if msg.image_md5:
                cache = await ImageCache.get_or_none(md5=msg.image_md5)
                if cache:
                    content = f"[系统注解: 图片内容为 {cache.caption}]\n{content}" if content else f"[系统注解: 图片内容为 {cache.caption}]"
            
            role = "assistant" if msg.role == "ai" else "user"
            message: ChatCompletionMessageParam = {
                "role": role,  # type: ignore
                "content": content
            }
            messages.append(message)
        
        return messages
