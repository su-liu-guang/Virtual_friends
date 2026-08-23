from datetime import datetime

from typing import List
from nonebot import logger
from openai.types.chat import ChatCompletionMessageParam
from .database import Message, Summary
from .logic import collect_message_image_blocks

class MemoryScheduler:
    """后台记忆整理调度器"""
    
    def __init__(self, chat_client):
        self.chat_client = chat_client
        self.processing_groups = set()
    
    async def check_and_process(self, group_id: str):
        """检查并处理积压消息"""
        if group_id in self.processing_groups:
            return
        
        pending_count = await Message.filter(
            group_id=group_id, 
            is_processed=False
        ).count()
        
        if pending_count < 50:
            return
        
        self.processing_groups.add(group_id)
        
        try:
            await self._process_batch(group_id)
            await self._check_archive(group_id)
        finally:
            self.processing_groups.discard(group_id)
    
    async def _process_batch(self, group_id: str):
        """处理一批消息"""
        messages = await Message.filter(
            group_id=group_id,
            is_processed=False
        ).order_by("timestamp").limit(50).all()
        
        if not messages:
            return
        
        image_blocks_by_message = await collect_message_image_blocks(
            messages,
            self.chat_client.file_cache_scope,
        )
        context_lines = []
        for msg in messages:
            role_label = "AI" if msg.role == "ai" else "用户"
            display_name = msg.user_nickname or role_label
            local_timestamp = msg.timestamp.astimezone()
            time_str = f"{local_timestamp.strftime('%Y-%m-%d %H:%M')} {msg.weekday}"
            content = msg.content or "(无文本内容)"
            if msg.id in image_blocks_by_message:
                content += f" [附图{len(image_blocks_by_message[msg.id])}张]"
            context_lines.append(f"[{time_str}] {display_name}: {content}")
        
        context = "\n".join(context_lines)
        
        # 生成摘要
        ordered_image_blocks = [
            block
            for msg in messages
            for block in image_blocks_by_message.get(msg.id, [])
        ]
        summary_ok = await self._generate_summary(
            group_id,
            context,
            messages,
            ordered_image_blocks,
        )

        # 标记已处理
        if summary_ok:
            message_ids = [msg.id for msg in messages]
            await Message.filter(id__in=message_ids).update(is_processed=True)
        else:
            logger.warning(
                f"[Scheduler] 本批处理未完成 (summary_ok={summary_ok})，保持未处理状态以便重试"
            )
    
    async def _generate_summary(
        self,
        group_id: str,
        context: str,
        messages: List,
        image_blocks: List,
    ) -> bool:
        """生成 L1 摘要，成功返回 True"""
        summary_text = await self.chat_client.generate_summary(
            context,
            image_blocks=image_blocks,
        )
        if not summary_text:
            logger.error(f"[Scheduler] 生成摘要失败，已跳过写入 (group={group_id})")
            return False
        
        start_time = messages[0].timestamp.astimezone()
        end_time = messages[-1].timestamp.astimezone()
        time_range = f"{start_time.strftime('%Y.%m.%d')}-{end_time.strftime('%m.%d')}"
        
        await Summary.create(
            group_id=group_id,
            level=1,
            content=summary_text,
            time_range=time_range
        )
        return True
    
    
    async def _check_archive(self, group_id: str):
        """检查并执行归档"""
        # L1 -> L2 归档（阈值放宽到 80）
        l1_count = await Summary.filter(
            group_id=group_id,
            level=1,
            is_archived=False
        ).count()
        
        if l1_count >= 80:
            await self._archive_summaries(group_id, 1, 2, 80)
        
        # L2 -> L3 归档（阈值放宽到 30）
        l2_count = await Summary.filter(
            group_id=group_id,
            level=2,
            is_archived=False
        ).count()
        
        if l2_count >= 30:
            await self._archive_summaries(group_id, 2, 3, 30)
    
    async def _archive_summaries(self, group_id: str, from_level: int, to_level: int, batch_size: int):
        """归档摘要到更高层级"""
        summaries = await Summary.filter(
            group_id=group_id,
            level=from_level,
            is_archived=False
        ).order_by("created_at").limit(batch_size).all()
        
        if not summaries:
            return
        
        combined_text = "\n\n".join([s.content for s in summaries])
        prompt = f"将以下多条摘要合并为一条更高层次的概括,保留关键信息:\n\n{combined_text}"
        
        messages: List[ChatCompletionMessageParam] = [{"role": "user", "content": prompt}]
        merged_summary = await self.chat_client.generate_response(messages, retry=3, temperature=0.5, thinking_mode="disabled")
        if not merged_summary:
            logger.error(f"[Scheduler] 合并摘要失败，已跳过写入 (group={group_id})")
            return
        
        start_time = summaries[0].time_range.split("-")[0]
        end_time = summaries[-1].time_range.split("-")[-1]
        time_range = f"{start_time}-{end_time}"
        
        await Summary.create(
            group_id=group_id,
            level=to_level,
            content=merged_summary,
            time_range=time_range
        )
        
        summary_ids = [s.id for s in summaries]
        await Summary.filter(id__in=summary_ids).update(is_archived=True)
