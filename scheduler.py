import asyncio
from datetime import datetime
from typing import List, Dict
from nonebot import logger
from openai.types.chat import ChatCompletionMessageParam
from .database import Message, Summary, ImportantEvent

class MemoryScheduler:
    """后台记忆整理调度器"""
    
    def __init__(self, chat_client):
        self.chat_client = chat_client
        self.processing_groups = set()
        self.last_fact_maintenance: Dict[str, datetime] = {}
    
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
        
        context_lines = []
        for msg in messages:
            role_label = "AI" if msg.role == "ai" else "用户"
            display_name = msg.user_nickname or role_label
            local_timestamp = msg.timestamp.astimezone()
            time_str = f"{local_timestamp.strftime('%Y-%m-%d %H:%M')} {msg.weekday}"
            content = msg.content or "(无文本内容)"
            context_lines.append(f"[{time_str}] {display_name}: {content}")
        
        context = "\n".join(context_lines)
        
        # 并行任务 A/B
        summary_task = self._generate_summary(group_id, context, messages)
        facts_task = self._extract_facts(group_id, context)

        summary_ok, facts_ok = await asyncio.gather(summary_task, facts_task)

        # 仅当摘要与事实均成功时才标记已处理
        if summary_ok and facts_ok:
            message_ids = [msg.id for msg in messages]
            await Message.filter(id__in=message_ids).update(is_processed=True)
        else:
            logger.warning(
                f"[Scheduler] 本批处理未完成 (summary_ok={summary_ok}, facts_ok={facts_ok})，保持未处理状态以便重试"
            )
    
    async def _generate_summary(self, group_id: str, context: str, messages: List) -> bool:
        """生成 L1 摘要，成功返回 True"""
        summary_text = await self.chat_client.generate_summary(context)
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
    
    async def _extract_facts(self, group_id: str, context: str) -> bool:
        """提取并存储重要事实（使用 v2 结构化提取），成功返回 True"""
        facts = await self.chat_client.extract_facts_v2(context)
        
        for fact_data in facts:
            content = f"{fact_data.get('person', '未知')}: {fact_data.get('content', '')}"
            await ImportantEvent.create(
                group_id=group_id,
                event_content=content,
                fact_type=fact_data.get("type"),
                confidence=fact_data.get("confidence"),
                recorded_date=datetime.now().date(),
            )
        return True
    
    async def _check_archive(self, group_id: str):
        """检查并执行归档 + 事实维护"""
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

        # Fact Maintenance: 事实数量 >= 30 时触发
        fact_count = await ImportantEvent.filter(
            group_id=group_id, validity=True
        ).count()

        if fact_count >= 30:
            now = datetime.now()
            last = self.last_fact_maintenance.get(group_id)
            if not last or (now - last).total_seconds() > 7200:
                self.last_fact_maintenance[group_id] = now
                asyncio.create_task(self._maintain_facts(group_id))
    
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
        merged_summary = await self.chat_client.generate_response(messages, retry=3, temperature=0.5, disable_thinking=True)
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

    async def _maintain_facts(self, group_id: str):
        """AI 审核事实库：过期标记 + 矛盾检测 + 同类合并"""
        all_facts = await ImportantEvent.filter(
            group_id=group_id, validity=True
        ).order_by("recorded_date").all()

        if len(all_facts) < 30:
            return

        facts_text = "\n".join(
            f"[{i+1}] ({f.recorded_date}) [{f.fact_type or 'unknown'}] {f.event_content}"
            for i, f in enumerate(all_facts)
        )

        today = datetime.now().date()
        prompt = f"""今天是 {today}。以下是当前全部有效事实 ({len(all_facts)} 条)：

{facts_text}

请完成以下三项任务：

1.【过期标记】标记已失效的 plan（日期已过的计划）和 commitment（承诺日期已过且无后续提及）
2.【矛盾检测】检测互相冲突的事实对（如"小明在上海工作"vs"小明已搬到北京"），给出更新建议
3.【同类合并】对主题相同但分散在多个 fact 中的信息，生成一条合并摘要（如 3 条关于"小明学日语"的 fact 合并为 1 条）

输出严格的 JSON 对象：
{{"expire": [失效编号列表], "conflicts": [{{"obsolete": 编号, "keep": 编号, "reason": "原因"}}], "merges": [{{"merge_ids": [编号列表], "merged_content": "合并后的完整事实"}}]}}"""

        result = await self.chat_client.extract_maintenance_actions(prompt)
        if not result:
            return

        # 执行过期标记
        for idx in result.get("expire", []):
            if 1 <= idx <= len(all_facts):
                all_facts[idx - 1].validity = False
                await all_facts[idx - 1].save()

        # 执行矛盾解决
        for conflict in result.get("conflicts", []):
            obs = conflict.get("obsolete")
            if obs and 1 <= obs <= len(all_facts):
                all_facts[obs - 1].validity = False
                await all_facts[obs - 1].save()

        # 执行合并
        for merge in result.get("merges", []):
            merged = merge.get("merged_content")
            if not merged:
                continue
            await ImportantEvent.create(
                group_id=group_id,
                event_content=f"[merged] {merged}",
                fact_type="merged",
                recorded_date=today,
            )
            for idx in merge.get("merge_ids", []):
                if isinstance(idx, int) and 1 <= idx <= len(all_facts):
                    all_facts[idx - 1].validity = False
                    await all_facts[idx - 1].save()

        expire_n = len(result.get("expire", []))
        conflict_n = len(result.get("conflicts", []))
        merge_n = len(result.get("merges", []))
        logger.success(
            f"[Fact Maintenance] 群 {group_id}: 过期 {expire_n}, 矛盾 {conflict_n}, 合并 {merge_n}"
        )
