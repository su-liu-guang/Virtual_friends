import random
import asyncio
import re
from datetime import datetime, timezone
from typing import List, Optional, Tuple

import nonebot
from nonebot import logger

from .config import ConfigManager
from .logic import ContextBuilder
from .clients import ChatClient
from .database import Message
from .scheduler import MemoryScheduler


WEEKDAY_CN = ["一", "二", "三", "四", "五", "六", "日"]


def truncate_to_minute(dt: datetime) -> datetime:
	return dt.replace(second=0, microsecond=0)


def weekday_label(dt: datetime) -> str:
	return f"星期{WEEKDAY_CN[dt.isoweekday() - 1]}"


class ActiveBehaviorManager:
	"""控制 AI 主动发言行为的调度器"""

	def __init__(
		self,
		config_manager: ConfigManager,
		chat_client: ChatClient,
		context_builder: ContextBuilder,
		memory_scheduler: MemoryScheduler,
		*,
		cooldown_hours: int = 2,
		idle_trigger_probability: float = 0.05,
	):
		self.config_manager = config_manager
		self.chat_client = chat_client
		self.context_builder = context_builder
		self.memory_scheduler = memory_scheduler
		self.cooldown_hours = cooldown_hours
		# 记录每个群组上次检查的时间
		self.last_checked: dict[str, datetime] = {}

	async def run_tick(self):
		"""周期性扫描全部群组,判断是否需要主动发言"""
		group_ids = self.config_manager.get_all_group_ids()
		if not group_ids:
			return

		# logger.debug(f"[Active] 正在扫描 {len(group_ids)} 个群组的主动行为条件")
		for group_id in group_ids:
			try:
				await self._process_group(group_id)
			except asyncio.CancelledError:
				raise
			except Exception as exc:  # pragma: no cover - 避免单个群异常影响全局
				logger.error(f"[Active] 处理群 {group_id} 时异常: {exc}", exc_info=True)

	async def _process_group(self, group_id: str):
		config = self.config_manager.get_instance_config(group_id)

		# 硬性过滤层
		if not config.get("active_mode", False):
			return
		if not self.config_manager.is_in_whitelist(group_id):
			return

		now = datetime.now().astimezone()
		
		# 检查间隔控制
		check_interval = config.get("active_check_interval", 45)  # 默认 45 分钟
		last_check = self.last_checked.get(group_id)
		if last_check:
			minutes_since = (now - last_check).total_seconds() / 60
			if minutes_since < check_interval:
				return
		
		# 更新检查时间
		self.last_checked[group_id] = now

		if not self._within_active_hours(config.get("active_hours"), now.hour):
			return

		# 软性过滤层
		last_message = await Message.filter(group_id=group_id).order_by("-timestamp").first()
		group_cooldown = self._safe_int(config.get("cooldown_hours"), default=self.cooldown_hours)
		hours_since_last_msg = self._hours_since(last_message.timestamp, now) if last_message else float("inf")
		if hours_since_last_msg < group_cooldown:
			logger.debug(
				f"[Active] 群 {group_id} 距离上次互动仅 {hours_since_last_msg:.2f}h, 仍在冷却中(冷却阈值: {group_cooldown}h)"
			)
			return

		last_user_message = await Message.filter(group_id=group_id, role="user").order_by("-timestamp").first()
		hours_since_user = self._hours_since(last_user_message.timestamp, now) if last_user_message else float("inf")

		silence_threshold = self._safe_int(config.get("silence_threshold"), default=24)
		is_revival_mode = hours_since_user >= silence_threshold

		trigger_prob = config.get("idle_trigger_probability", 0.05)
		if not is_revival_mode and random.random() > trigger_prob:
			logger.debug(f"[Active] 群 {group_id} 未命中概率触发({trigger_prob}), 本轮跳过")
			return

		await self._send_active_message(
			group_id=group_id,
			config=config,
			now=now,
			is_revival_mode=is_revival_mode,
			hours_since_user=hours_since_user,
		)

	async def _send_active_message(
		self,
		group_id: str,
		config: dict,
		now: datetime,
		is_revival_mode: bool,
		hours_since_user: float,
	):
		context = await self.context_builder.build_context(
			group_id=group_id,
			user_nickname="群友",
			current_time=now,
		)

		prompt = self._build_prompt(is_revival_mode, hours_since_user)
		context.append({"role": "user", "content": prompt})

		reply = (await self.chat_client.generate_response(context)).strip()
		
		# 去除可能的时间戳前缀 [2025-12-12 19:30]
		reply = re.sub(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]\s*", "", reply)
		
		if not reply:
			logger.warning(f"[Active] 群 {group_id} 生成的主动消息为空, 已忽略")
			return

		sent, sender_id = await self._deliver_message(group_id, reply)
		if sent:
			stored_time = now
			display_time = now.strftime("%Y-%m-%d %H:%M")
			await Message.create(
				group_id=group_id,
				role="ai",
				content=reply,
				user_nickname=None,
				user_id=sender_id,
				timestamp=stored_time,
				display_time=display_time,
				weekday=weekday_label(stored_time),
				is_processed=False,
			)
			logger.success(
				f"[Active] 已向群 {group_id} 发送{'唤醒' if is_revival_mode else '闲聊'}消息: {reply[:30]}..."
			)
			
			# 触发后台记忆整理
			asyncio.create_task(self.memory_scheduler.check_and_process(group_id))
		else:
			logger.warning(f"[Active] 向群 {group_id} 发送主动消息失败")

	async def _deliver_message(self, group_id: str, content: str) -> Tuple[bool, Optional[str]]:
		driver = nonebot.get_driver()
		if not driver.bots:
			logger.warning("[Active] 当前无在线 Bot, 无法主动发言")
			return False, None

		for bot in driver.bots.values():
			try:
				target = int(group_id)
			except ValueError:
				target = group_id

			try:
				await bot.call_api("send_group_msg", group_id=target, message=content)
				return True, str(bot.self_id)
			except Exception as exc:
				logger.debug(f"[Active] Bot({bot.self_id}) 发送失败, 尝试下一个: {exc}")
				continue
		return False, None

	@staticmethod
	def _hours_since(timestamp: datetime, now: datetime) -> float:
		# 确保存储的时间戳转换为本地时间进行比较
		if timestamp.tzinfo is None:
			# 如果是 naive 时间，假设它是 UTC (Tortoise 默认) 并转换为本地
			# 或者如果项目约定 naive 就是本地，则直接加上本地时区
			# 这里为了稳妥，先将其视为 UTC (如果数据库存的是 UTC)
			# 但通常 Tortoise 取出的 DatetimeField 是 aware 的 (如果配置了时区)
			# 最安全的做法是统一转为 astimezone()
			ts = timestamp.replace(tzinfo=timezone.utc).astimezone()
		else:
			ts = timestamp.astimezone()
			
		delta = now - ts
		return delta.total_seconds() / 3600

	@staticmethod
	def _safe_int(value: Optional[object], default: int) -> int:
		try:
			return int(value)
		except (TypeError, ValueError):
			return default

	def _within_active_hours(self, active_hours: Optional[List[int]], current_hour: int) -> bool:
		if not active_hours or len(active_hours) != 2:
			return True
		try:
			start = int(active_hours[0]) % 24
			end = int(active_hours[1]) % 24
		except (TypeError, ValueError):
			return True

		if start == end:
			return True  # 等于视为全天开启

		if start < end:
			return start <= current_hour < end
		return current_hour >= start or current_hour < end

	def _build_prompt(self, is_revival_mode: bool, hours_since_user: float) -> str:
		if is_revival_mode:
			base = """[系统指令: 主动唤醒模式]
检测到当前对话已中断较长时间（冷场）。
请根据【记忆回顾】和【已知事实】，主动发起一个新的话题来打破沉默。
你可以：
1. 关心用户的近况（如忙碌程度、心情）。
2. 追问之前对话中未完结的事件（如“之前的感冒好了吗？”“那个项目怎么样了？”）。
3. 如果没有旧话题，就分享一个轻松的生活化话题。
要求：语气自然亲切，像好久不见的朋友打招呼。禁止使用“在吗”、“你好”等机械开场。"""
		else:
			base = """[系统指令: 日常闲聊模式]
现在是闲暇时间，请根据【当前时间】和【近期对话】，发起一个轻松的短话题。
请自行判断当前时间点（早安/午饭/摸鱼/深夜）：
- 如果是清晨，可以问候早安或询问计划。
- 如果是饭点，可以聊聊食物。
- 如果是深夜，提醒休息或聊聊感性话题。
- 其他时间，可以分享趣闻、吐槽或接着上文的话题延伸。
要求：简短、有趣、不唐突。不要重复刚刚说过的话。"""

		silence_text = (
			"\n(系统提示: 距离上次用户发言约 %.1f 小时)" % hours_since_user
			if hours_since_user != float("inf")
			else ""
		)

		return f"{base}{silence_text}"

