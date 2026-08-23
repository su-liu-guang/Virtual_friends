from typing import List
from nonebot import logger
from tortoise.transactions import in_transaction
from .database import Message, Summary
from .logic import collect_message_image_blocks


L1_ARCHIVE_BATCH_SIZE = 20
L2_ARCHIVE_BATCH_SIZE = 10

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
        ).order_by("timestamp", "id").limit(50).all()
        
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
            group_id=group_id,
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
        for from_level, to_level, batch_size in (
            (1, 2, L1_ARCHIVE_BATCH_SIZE),
            (2, 3, L2_ARCHIVE_BATCH_SIZE),
        ):
            while True:
                count = await Summary.filter(
                    group_id=group_id,
                    level=from_level,
                    is_archived=False,
                ).count()
                if count < batch_size:
                    break
                if not await self._archive_summaries(
                    group_id, from_level, to_level, batch_size
                ):
                    break
    
    async def _archive_summaries(
        self,
        group_id: str,
        from_level: int,
        to_level: int,
        batch_size: int,
    ) -> bool:
        """归档摘要到更高层级，成功返回 True。"""
        summaries = await Summary.filter(
            group_id=group_id,
            level=from_level,
            is_archived=False
        ).order_by("created_at", "id").limit(batch_size).all()
        
        if not summaries:
            return False
        
        combined_text = "\n\n".join([s.content for s in summaries])
        merged_summary = await self.chat_client.generate_archive_summary(
            combined_text,
            group_id=group_id,
            from_level=from_level,
            to_level=to_level,
        )
        if not merged_summary:
            logger.error(f"[Scheduler] 合并摘要失败，已跳过写入 (group={group_id})")
            return False
        
        start_time = summaries[0].time_range.split("-")[0]
        end_time = summaries[-1].time_range.split("-")[-1]
        time_range = f"{start_time}-{end_time}"
        
        summary_ids = [s.id for s in summaries]
        async with in_transaction() as connection:
            await Summary.create(
                group_id=group_id,
                level=to_level,
                content=merged_summary,
                time_range=time_range,
                using_db=connection,
            )
            updated = await Summary.filter(
                id__in=summary_ids,
                is_archived=False,
            ).using_db(connection).update(is_archived=True)
            if updated != len(summary_ids):
                raise RuntimeError(
                    f"归档摘要并发冲突: expected={len(summary_ids)}, updated={updated}"
                )
        logger.success(
            f"[Scheduler] L{from_level}→L{to_level} 完成: "
            f"group={group_id}, source={len(summary_ids)}, chars={len(merged_summary)}"
        )
        return True
