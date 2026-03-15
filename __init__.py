import nonebot
from nonebot import on_message, on_command, logger, require
from nonebot.adapters.onebot.v11 import Bot, MessageEvent, GroupMessageEvent, Message as OB11Message
from nonebot.adapters.onebot.v11.permission import GROUP_OWNER,GROUP_ADMIN
from nonebot.permission import SUPERUSER
from nonebot.exception import MatcherException, FinishedException
from nonebot.params import CommandArg
from nonebot.plugin import PluginMetadata
import json
from datetime import datetime, timedelta
import asyncio
import random
import re
from pathlib import Path

from .config import ConfigManager
from .database import init_db, Message, Summary, ImportantEvent, MemoryVector
from .clients import VisionClient, ChatClient, EmbeddingClient
from .logic import (
    ContextBuilder,
    process_image_message,
    sanitize_persona_reply,
    has_complete_persona_reply_tag,
    PERSONA_REPLY_RETRY_PROMPT,
)
from .scheduler import MemoryScheduler
from .active_behavior import ActiveBehaviorManager
from .summary import DailySummaryGenerator
from .knowledge import KnowledgeBase
from .memory_retriever import MemoryRetriever

__plugin_meta__ = PluginMetadata(
    name="Virtual Friends",
    description="虚拟好友群聊插件，提供人设驱动对话与记忆管理能力。",
    usage="发送 /vf帮助 查看所有功能指令。",
    type="application",
    supported_adapters={"~onebot.v11"},
)

# ================= 全局组件与客户端 =================
config_manager = ConfigManager()
vision_client = VisionClient()
chat_client = ChatClient()
embedding_client = EmbeddingClient()
knowledge_base = KnowledgeBase()
memory_retriever = MemoryRetriever(embedding_client, config_manager)
context_builder = ContextBuilder(config_manager, knowledge_base, memory_retriever)
memory_scheduler = MemoryScheduler(chat_client, memory_retriever)
active_behavior_manager = ActiveBehaviorManager(config_manager, chat_client, context_builder, memory_scheduler)
daily_summary_generator = DailySummaryGenerator(chat_client, config_manager)

try:
    scheduler = require("nonebot_plugin_apscheduler").scheduler
except RuntimeError:
    scheduler = None
    logger.warning("未安装 nonebot_plugin_apscheduler, 主动行为功能已禁用")

# ================= 常量与别名 =================

WEEKDAY_CN = ["一", "二", "三", "四", "五", "六", "日"]

# 配置键名别名映射（内部 key -> 展示名）
KEY_ALIAS = {
    "persona_name": "人设名称",
    "reply_rate": "被动回复概率",
    "active_mode": "主动发言开关",
    "active_hours": "主动发言时间段 [起始小时, 结束小时]",
    "cooldown_hours": "主动发言冷却(小时)",
    "active_check_interval": "主动检查间隔(分钟)",
    "idle_trigger_probability": "闲聊触发概率",
    "silence_threshold": "沉默阈值(小时)",
    "group_name": "群名称",
    "summary_enabled": "每日总结开关",
    "summary_time": "每日总结时间",
}

# 展示名/中文别名 -> 内部 key
ALIAS_TO_KEY = {
    # persona_name
    "人设名称": "persona_name",
    "人设": "persona_name",
    "persona_name": "persona_name",
    # reply_rate
    "被动回复概率": "reply_rate",
    "reply_rate": "reply_rate",
    # active_mode
    "主动发言开关": "active_mode",
    "active_mode": "active_mode",
    # active_hours
    "主动发言时间段": "active_hours",
    "active_hours": "active_hours",
    # cooldown_hours
    "主动发言冷却(小时)": "cooldown_hours",
    "冷却时间": "cooldown_hours",
    "cooldown_hours": "cooldown_hours",
    # active_check_interval
    "主动检查间隔(分钟)": "active_check_interval",
    "active_check_interval": "active_check_interval",
    # idle_trigger_probability
    "闲聊触发概率": "idle_trigger_probability",
    "idle_trigger_probability": "idle_trigger_probability",
    # silence_threshold
    "沉默阈值(小时)": "silence_threshold",
    "silence阈值": "silence_threshold",
    "silence_threshold": "silence_threshold",
    # group_name
    "群名称备注": "group_name",
    "group_name": "group_name",
    # summary_enabled
    "每日总结开关": "summary_enabled",
    "summary_enabled": "summary_enabled",
    "开启每日总结": "summary_enabled",
    # summary_time
    "每日总结时间": "summary_time",
    "summary_time": "summary_time",
    "总结时间": "summary_time",
}

# ================= 命令前缀与工具函数 =================

DRIVER = nonebot.get_driver()
COMMAND_PREFIXES = tuple(prefix for prefix in DRIVER.config.command_start if prefix)

def is_command_like_message(text: str) -> bool:
    stripped = text.lstrip()
    if not stripped or not COMMAND_PREFIXES:
        return False
    return stripped.startswith(COMMAND_PREFIXES)


def format_context_for_debug(context: list) -> str:
    lines = ["[VF Debug] 发送给 AI 的完整上下文:"]
    for idx, msg in enumerate(context, start=1):
        role = msg.get("role", "unknown")
        content = msg.get("content", "")
        if not isinstance(content, str):
            try:
                content = json.dumps(content, ensure_ascii=False)
            except TypeError:
                content = str(content)
        lines.append(f"[{idx}] role={role}\n{content}")
    return "\n".join(lines)

# ================= 工具函数 =================

def get_group_id(event: MessageEvent) -> str:
    return str(event.group_id) if isinstance(event, GroupMessageEvent) else str(event.user_id)


def get_current_time() -> datetime:
    """获取当前带时区的精准时间"""
    return datetime.now().astimezone()


def format_display_time(dt: datetime) -> str:
    """格式化为人类/AI易读的分钟级时间"""
    return dt.strftime("%Y-%m-%d %H:%M")


def get_weekday_label(dt: datetime) -> str:
    return f"星期{WEEKDAY_CN[dt.isoweekday() - 1]}"

# ================= 消息处理 =================

message_handler = on_message(priority=95, block=False)

@message_handler.handle()
async def handle_message(bot: Bot, event: MessageEvent):
    # 过滤机器人自己发送的消息
    if str(event.user_id) == str(bot.self_id):
        return
    
    group_id = get_group_id(event)
    text_content = event.get_plaintext().strip()
    user_nickname = event.sender.card or event.sender.nickname or f"用户{event.user_id}"
    user_id = str(event.user_id)

    if is_command_like_message(text_content):
        return
    
    # 白名单检查
    if isinstance(event, GroupMessageEvent) and not config_manager.is_in_whitelist(group_id):
        return
    
    # 加载配置
    config = config_manager.get_instance_config(group_id)
    
    # 回复频率判断
    reply_rate = config.get("reply_rate", 0.3)
    at_bot = event.is_tome()
    should_reply = at_bot or random.random() < reply_rate
    has_image = any(seg.type == "image" for seg in event.message)

    if has_image and not should_reply:
        return

    logger.info(f"收到消息 [群组: {group_id}] [用户: {event.user_id}]: {text_content[:50]}...")
    
    image_md5 = None
    if has_image:
        for seg in event.message:
            if seg.type == "image":
                image_url = seg.data["url"]
                summary = seg.data.get("summary", "")
                is_sticker = bool(summary)
                logger.debug(f"检测到图片: {image_url[:50]}... (Summary: {summary})")
                image_md5 = await process_image_message(image_url, vision_client, is_sticker=is_sticker)
                logger.debug(f"图片处理完成, MD5: {image_md5}")
                break
    
    timestamp = get_current_time()
    display_time = format_display_time(timestamp)
    weekday_label = get_weekday_label(timestamp)
    
    await Message.create(
        group_id=group_id,
        role="user",
        content=text_content,
        user_nickname=user_nickname,
        user_id=user_id,
        image_md5=image_md5,
        timestamp=timestamp,
        display_time=display_time,
        weekday=weekday_label,
        is_processed=False
    )
    

    
    if not should_reply:
        asyncio.create_task(memory_scheduler.check_and_process(group_id))
        return
    
    # 构建上下文并生成回复
    context = await context_builder.build_context(
        group_id=group_id,
        user_nickname=user_nickname,
        current_time=timestamp
    )
    logger.debug(format_context_for_debug(context))
    response_raw = await chat_client.generate_response(context)
    if not response_raw:
        logger.error("[Chat] 生成回复失败，已跳过存储与发送")
        return

    if not has_complete_persona_reply_tag(response_raw):
        logger.warning("[Chat] 首次回复标签不完整，尝试一次格式修复重试")
        repair_context = [
            *context,
            {"role": "user", "content": PERSONA_REPLY_RETRY_PROMPT},
        ]
        repaired = await chat_client.generate_response(repair_context, retry=1)
        if repaired:
            response_raw = repaired

    response = sanitize_persona_reply(response_raw)
    if not response:
        logger.error("[Chat] 生成回复为空，已跳过存储与发送")
        return
    
    # 存储并发送回复
    reply_timestamp = get_current_time()
    reply_display_time = format_display_time(reply_timestamp)
    
    await Message.create(
        group_id=group_id,
        role="ai",
        content=response,
        user_nickname=None,
        user_id=str(bot.self_id),
        timestamp=reply_timestamp,
        display_time=reply_display_time,
        weekday=get_weekday_label(reply_timestamp),
        is_processed=False
    )
    
    # 处理图片发送指令
    # 格式: {{发送图片:xxx}}
    from nonebot.adapters.onebot.v11 import MessageSegment
    
    final_message = OB11Message(response)
    
    img_match = re.search(r"\{\{发送图片:(.*?)\}\}", response)
    if img_match:
        img_name = img_match.group(1)
        img_path = knowledge_base.get_image_path(img_name)
        
        # 移除指令文本
        clean_response = response.replace(img_match.group(0), "").strip()
        final_message = OB11Message(clean_response)
        
        if img_path:
            try:
                final_message += MessageSegment.image(Path(img_path))
                logger.info(f"附加图片: {img_name} -> {img_path}")
            except Exception as e:
                logger.error(f"图片加载失败: {e}")
        else:
            logger.warning(f"未找到图片: {img_name}")

    await message_handler.send(final_message)
    
    # 触发后台任务
    asyncio.create_task(memory_scheduler.check_and_process(group_id))

# ================= 指令注册 =================

switch_persona = on_command("切换人设", aliases={"切换提示词"}, priority=5, block=True, permission=SUPERUSER | GROUP_OWNER | GROUP_ADMIN)
memory_status = on_command("记忆状态", aliases={"查看记忆"}, priority=5, block=True)
forget_cmd = on_command("遗忘", priority=5, block=True, permission=SUPERUSER | GROUP_OWNER | GROUP_ADMIN)
clear_memory = on_command("清空记忆", aliases={"重置记忆"}, priority=5, block=True, permission=SUPERUSER | GROUP_OWNER | GROUP_ADMIN)
persona_list = on_command("提示词列表", aliases={"人设列表"}, permission=SUPERUSER | GROUP_OWNER | GROUP_ADMIN, priority=5, block=True)
view_persona = on_command("查看提示词", aliases={"查看人设"}, priority=5, block=True)
add_persona = on_command("添加提示词", aliases={"添加人设", "增加提示词", "增加人设"}, priority=5, block=True, permission=SUPERUSER | GROUP_OWNER | GROUP_ADMIN)
delete_persona = on_command("删除提示词", aliases={"删除人设"}, priority=5, block=True, permission=SUPERUSER)
enable_plugin = on_command("人生启动", aliases={"世界开启", "故事开始"}, priority=5, block=True, permission=SUPERUSER | GROUP_OWNER | GROUP_ADMIN)
disable_plugin = on_command("世界终结", priority=5, block=True, permission=SUPERUSER | GROUP_OWNER | GROUP_ADMIN)
view_whitelist = on_command("查看白名单", priority=5, block=True, permission=SUPERUSER)
view_config = on_command("查看配置", aliases={"vf配置", "当前配置"}, priority=5, block=True, permission=SUPERUSER | GROUP_OWNER | GROUP_ADMIN)
update_config = on_command("修改配置", aliases={"vf设置", "更新配置"}, priority=5, block=True, permission=SUPERUSER | GROUP_OWNER | GROUP_ADMIN)
reload_config = on_command("重载配置", aliases={"刷新配置", "重载人设", "刷新人设"}, priority=5, block=True, permission=SUPERUSER | GROUP_OWNER | GROUP_ADMIN)
daily_summary_cmd = on_command("今日总结", aliases={"群聊日报"}, priority=5, block=True)
test_summary_cmd = on_command("测试日报", aliases={"历史日报"}, priority=5, block=True, permission=SUPERUSER)
yesterday_summary_cmd = on_command("昨日总结", aliases={"昨天总结", "昨日日报"}, priority=5, block=True)
help_cmd = on_command("vf帮助", aliases={"vf菜单", "vf指令列表"}, priority=5, block=True)
backfill_vectors_cmd = on_command("回填记忆向量", priority=5, block=True, permission=SUPERUSER)

# ================= 指令处理 =================

# --- 人设与记忆 ---

@switch_persona.handle()
async def handle_switch_persona(bot: Bot, event: MessageEvent, args: OB11Message = CommandArg()):
    arg_text = args.extract_plain_text().strip()
    if not arg_text:
        await switch_persona.finish("用法: /切换人设 [名称]")
    
    group_id = get_group_id(event)
    personas = config_manager.get_personas()
    
    if arg_text not in personas:
        await switch_persona.finish(f"人设不存在。可用人设: {', '.join(personas.keys())}")
    
    config_manager.update_instance_config(group_id, {"persona_name": arg_text})
    await switch_persona.finish(f"已切换至人设: {arg_text}")

@memory_status.handle()
async def handle_memory_status(bot: Bot, event: MessageEvent):
    group_id = get_group_id(event)
    
    l1 = await Summary.filter(group_id=group_id, level=1, is_archived=False).count()
    l2 = await Summary.filter(group_id=group_id, level=2, is_archived=False).count()
    l3 = await Summary.filter(group_id=group_id, level=3, is_archived=False).count()
    facts = await ImportantEvent.filter(group_id=group_id, validity=True).count()
    pending = await Message.filter(group_id=group_id, is_processed=False).count()
    
    await memory_status.finish(f"""📊 记忆系统状态
━━━━━━━━━━━━━━
🧠 流式记忆:
  L1 细节摘要: {l1} 条
  L2 叙事概括: {l2} 条
  L3 宏观印象: {l3} 条

⭐ 锚点事实: {facts } 条
⏳ 待处理消息: {pending} 条
━━━━━━━━━━━━━━""")

@forget_cmd.handle()
async def handle_forget(bot: Bot, event: MessageEvent, args: OB11Message = CommandArg()):
    keyword = args.extract_plain_text().strip()
    if not keyword:
        await forget_cmd.finish("用法: /遗忘 [关键词]")
    
    group_id = get_group_id(event)
    deleted_count = await ImportantEvent.filter(group_id=group_id, event_content__contains=keyword).delete()
    
    if deleted_count:
        await forget_cmd.finish(f"已删除 {deleted_count} 条包含 '{keyword}' 的记忆")
    else:
        await forget_cmd.finish(f"未找到包含 '{keyword}' 的记忆")

@clear_memory.handle()
async def handle_clear_memory(bot: Bot, event: MessageEvent):
    group_id = get_group_id(event)
    try:
        await Message.filter(group_id=group_id).delete()
        await Summary.filter(group_id=group_id).delete()
        await ImportantEvent.filter(group_id=group_id).delete()
        await MemoryVector.filter(group_id=group_id).delete()
        await clear_memory.finish("✅ 已清空当前群组的所有记忆")
    except MatcherException:
        raise
    except Exception as e:
        await clear_memory.finish(f"❌ 清空失败: {e}")

# --- 人设配置 ---

@persona_list.handle()
async def handle_persona_list(bot: Bot, event: MessageEvent):
    group_id = get_group_id(event)
    personas = config_manager.get_personas()
    current_persona = config_manager.get_instance_config(group_id).get("persona_name")
    msg = "\n".join([f"• {k}" for k in personas.keys()])
    await persona_list.finish(
        f"🪪 当前人设: {current_persona}\n\n📝 可用人设:\n{msg}\n\n使用 /切换人设 [名称] 切换"
    )

@view_persona.handle()
async def handle_view_persona(bot: Bot, event: MessageEvent, args: OB11Message = CommandArg()):
    arg_text = args.extract_plain_text().strip()
    group_id = get_group_id(event)
    
    if not arg_text:
        arg_text = config_manager.get_instance_config(group_id).get("persona_name", "default")
    
    persona = config_manager.get_personas().get(arg_text)
    if not persona:
        await view_persona.finish(f"人设 '{arg_text}' 不存在")
        
    await view_persona.finish(f"📋 {arg_text}\n\n{persona.get('prompt')}")

@add_persona.handle()
async def handle_add_persona(bot: Bot, event: MessageEvent, args: OB11Message = CommandArg()):
    text = args.extract_plain_text().strip()
    parts = text.split(maxsplit=1)
    
    if len(parts) < 2:
        await add_persona.finish("用法: /添加提示词 名称 提示词")
    
    name, prompt = parts[0], parts[1]
    if config_manager.add_persona(name, prompt):
        await add_persona.finish(f"✅ 已添加人设 '{name}'")
    else:
        await add_persona.finish("❌ 添加失败")

@delete_persona.handle()
async def handle_delete_persona(bot: Bot, event: MessageEvent, args: OB11Message = CommandArg()):
    arg_text = args.extract_plain_text().strip()
    if not arg_text:
        await delete_persona.finish("用法: /删除提示词 [名称]")
    
    if arg_text == "default":
        await delete_persona.finish("❌ 不能删除默认人设")
        
    if config_manager.delete_persona(arg_text):
        await delete_persona.finish(f"✅ 已删除人设 '{arg_text}'")
    else:
        await delete_persona.finish("❌ 删除失败")

# --- 插件开关与白名单 ---

@enable_plugin.handle()
async def handle_enable_plugin(bot: Bot, event: GroupMessageEvent):
    group_id = str(event.group_id)
    # 获取群名用于记录
    try:
        group_info = await bot.get_group_info(group_id=int(group_id))
        group_name = group_info.get("group_name", "未知群组")
    except:
        group_name = "未知群组"

    if config_manager.add_to_whitelist(group_id, group_name):
        await enable_plugin.finish(f"✨ 人生启动成功！本群 ({group_name}) 已启用插件")
    else:
        await enable_plugin.finish("✅ 本群已启用")

@disable_plugin.handle()
async def handle_disable_plugin(bot: Bot, event: GroupMessageEvent):
    group_id = str(event.group_id)
    if config_manager.remove_from_whitelist(group_id):
        await disable_plugin.finish("🌙 世界终结... 本群已禁用插件")
    else:
        await disable_plugin.finish("⚠️ 本群未启用")

@view_whitelist.handle()
async def handle_view_whitelist(bot: Bot, event: MessageEvent):
    whitelist = config_manager.get_whitelist()
    if not whitelist:
        await view_whitelist.finish("📋 白名单为空")
        return

    msg_lines = []
    for gid in whitelist:
        cfg = config_manager.get_instance_config(gid)
        name = cfg.get("group_name", "未知群组")
        msg_lines.append(f"• {name} ({gid})")
    
    await view_whitelist.finish(f"📋 白名单群组:\n" + "\n".join(msg_lines))

# --- 配置管理 ---

@view_config.handle()
async def handle_view_config(bot: Bot, event: MessageEvent):
    group_id = get_group_id(event)
    if not config_manager.is_in_whitelist(group_id):
        await view_config.finish("⚠️ 本群未启用插件")
    
    config = config_manager.get_instance_config(group_id)
    # 复制一份配置，避免修改原对象
    display_config = config.copy()

    aliased_config = {}
    for k, v in display_config.items():
        if k == "whitelisted":
            continue
        aliased_key = KEY_ALIAS.get(k, k)
        aliased_config[aliased_key] = v
    
    await view_config.finish(
        "⚙️ 当前配置 \n /修改配置 \n"
        f"{json.dumps(aliased_config, ensure_ascii=False, indent=2)}"
    )

@update_config.handle()
async def handle_update_config(bot: Bot, event: MessageEvent, args: OB11Message = CommandArg()):
    group_id = get_group_id(event)
    if not config_manager.is_in_whitelist(group_id):
        await update_config.finish("⚠️ 本群未启用插件")
        
    content = args.extract_plain_text().strip()
    if not content:
        await update_config.finish("用法: /修改配置 <JSON数据>")
        
    try:
        new_config = json.loads(content)
    except Exception as e:
        await update_config.finish(f"❌ JSON 解析失败: {e}")
        
    if not isinstance(new_config, dict):
        await update_config.finish("❌ 配置必须是 JSON 对象")
        
    # 安全过滤：只允许修改特定的配置项（通过别名映射到内部 key）
    safe_config = {}
    valid_internal_keys = {
        "persona_name",
        "reply_rate",
        "active_mode",
        "active_hours",
        "cooldown_hours",
        "active_check_interval",
        "idle_trigger_probability",
        "silence_threshold",
        "group_name",
        "summary_enabled",
        "summary_time",
    }

    for raw_key, v in new_config.items():
        internal_key = ALIAS_TO_KEY.get(raw_key, raw_key)
        if internal_key in valid_internal_keys:
            safe_config[internal_key] = v
            
    if not safe_config:
        await update_config.finish("⚠️ 未检测到有效的配置项")
        
    config_manager.update_instance_config(group_id, safe_config)
    
    # 反馈更新后的配置
    final_config = config_manager.get_instance_config(group_id)
    # 同样移除 whitelisted，并应用别名展示
    display_final = final_config.copy()
    if "whitelisted" in display_final:
        del display_final["whitelisted"]

    aliased_final = {}
    for k, v in display_final.items():
        aliased_key = KEY_ALIAS.get(k, k)
        aliased_final[aliased_key] = v
        
    await update_config.finish(f"✅ 配置已更新:\n{json.dumps(aliased_final, ensure_ascii=False, indent=2)}")

@reload_config.handle()
async def handle_reload_config(bot: Bot, event: MessageEvent):
    success = await config_manager.reload()
    if success:
        await reload_config.finish("✅ 配置已重新加载！")
    else:
        await reload_config.finish("❌ 配置重载失败，请检查日志。")

@daily_summary_cmd.handle()
async def handle_daily_summary(bot: Bot, event: MessageEvent):
    group_id = get_group_id(event)
    await daily_summary_cmd.send("正在生成今日群聊日报，请稍候...")
    
    try:
        from nonebot.adapters.onebot.v11 import MessageSegment
        img = await daily_summary_generator.generate_report(group_id)
        if img:
            await daily_summary_cmd.finish(MessageSegment.image(img))
        else:
            await daily_summary_cmd.finish("生成日报失败，可能是今日无消息或缺少依赖。")
    except FinishedException:
        raise
    except Exception as e:
        logger.error(f"生成日报出错: {e}")
        await daily_summary_cmd.finish(f"生成日报出错: {e}")

@yesterday_summary_cmd.handle()
async def handle_yesterday_summary(bot: Bot, event: MessageEvent):
    group_id = get_group_id(event)
    target_date = datetime.now().astimezone() - timedelta(days=1)
    await yesterday_summary_cmd.send("正在生成昨日群聊日报，请稍候...")

    try:
        from nonebot.adapters.onebot.v11 import MessageSegment
        img = await daily_summary_generator.generate_report(group_id, target_date=target_date)
        if img:
            await yesterday_summary_cmd.finish(MessageSegment.image(img))
        else:
            await yesterday_summary_cmd.finish("生成昨日日报失败，可能是昨日无消息或缺少依赖。")
    except FinishedException:
        raise
    except Exception as e:
        logger.error(f"生成昨日日报出错: {e}")
        await yesterday_summary_cmd.finish(f"生成昨日日报出错: {e}")

@test_summary_cmd.handle()
async def handle_test_summary(bot: Bot, event: MessageEvent, args: OB11Message = CommandArg()):
    arg_text = args.extract_plain_text().strip()
    if not arg_text:
        await test_summary_cmd.finish("用法: /测试日报 [日期 YYYY-MM-DD] [可选群号]")
    
    parts = arg_text.split()
    date_str = parts[0]
    
    try:
        target_date = datetime.strptime(date_str, "%Y-%m-%d")
    except ValueError:
        await test_summary_cmd.finish("日期格式错误，请使用 YYYY-MM-DD 格式，例如 2023-12-11")
        
    if len(parts) > 1:
        group_id = parts[1]
    else:
        group_id = get_group_id(event)
        
    await test_summary_cmd.send(f"正在生成群 {group_id} 在 {date_str} 的群聊日报，请稍候...")
    
    try:
        from nonebot.adapters.onebot.v11 import MessageSegment
        img = await daily_summary_generator.generate_report(group_id, target_date=target_date)
        if img:
            await test_summary_cmd.finish(MessageSegment.image(img))
        else:
            await test_summary_cmd.finish("生成日报失败，可能是当日无消息或缺少依赖。")
    except FinishedException:
        raise
    except Exception as e:
        logger.error(f"生成测试日报出错: {e}")
        await test_summary_cmd.finish(f"生成测试日报出错: {e}")

# --- 帮助信息 ---

@help_cmd.handle()
async def handle_help(bot: Bot, event: MessageEvent):
    await help_cmd.finish("""🤖 Virtual Friends 指令列表
━━━━━━━━━━━━━━
基础指令:
• /查看提示词 [名称] - 查看指定人设详情
• /记忆状态 - 查看当前群聊的记忆统计
• /今日总结 - 生成今日群聊日报
• /昨日总结 - 生成昨日群聊日报

管理指令 (超管/群主/管理员):
• /提示词列表 - 查看所有可用人设
• /切换人设 [名称] - 切换当前群聊的 AI 人设
• /添加提示词 <名称> <内容> - 添加新人设
• /人生启动 - 在当前群启用插件
• /世界终结 - 在当前群禁用插件
• /清空记忆 - 删除当前群的所有记忆数据
• /遗忘 [关键词] - 删除包含关键词的特定记忆
• /查看配置 - 查看当前群组的详细配置
• /修改配置 <JSON> - 修改当前群组配置
• /重载配置 - 重新加载配置文件

超管指令:
• /删除提示词 [名称] - 删除指定人设
• /查看白名单 - 查看已启用插件的群
• /测试日报 [日期] - 生成指定日期的日报
• /回填记忆向量 - 补写历史摘要/事实的向量""")


@backfill_vectors_cmd.handle()
async def handle_backfill_vectors(bot: Bot, event: MessageEvent):
    if not memory_retriever:
        await backfill_vectors_cmd.finish("❌ 未启用记忆向量功能")

    await backfill_vectors_cmd.send("⏳ 开始回填历史摘要与事实向量，请稍候...")

    # 回填 Summary
    summary_count = 0
    last_id = 0
    while True:
        batch = await Summary.filter(id__gt=last_id).order_by("id").limit(100).all()
        if not batch:
            break
        for s in batch:
            await memory_retriever.upsert_summary(s)
        summary_count += len(batch)
        last_id = batch[-1].id

    # 回填 ImportantEvent
    fact_count = 0
    last_fact_id = 0
    while True:
        batch = await ImportantEvent.filter(id__gt=last_fact_id).order_by("id").limit(100).all()
        if not batch:
            break
        for f in batch:
            await memory_retriever.upsert_fact(f)
        fact_count += len(batch)
        last_fact_id = batch[-1].id

    await backfill_vectors_cmd.finish(f"✅ 回填完成：摘要 {summary_count} 条，事实 {fact_count} 条")


# ================= 生命周期事件 =================

driver = DRIVER

async def check_daily_summary():
    """检查并发送每日总结"""
    now = datetime.now()
    current_time_str = now.strftime("%H:%M")
    
    # 获取所有群组配置
    for group_id, config in config_manager.groups.items():
        if not config.get("summary_enabled"):
            continue
            
        target_time = config.get("summary_time", "23:00")
        if target_time == current_time_str:
            logger.info(f"触发群 {group_id} 每日总结")
            try:
                # 尝试获取 bot 实例
                bot = nonebot.get_bot()
                from nonebot.adapters.onebot.v11 import MessageSegment
                img = await daily_summary_generator.generate_report(group_id)
                if img:
                    await bot.send_group_msg(group_id=int(group_id), message=MessageSegment.image(img))
            except Exception as e:
                logger.error(f"发送群 {group_id} 每日总结失败: {e}")

@driver.on_startup
async def startup():
    await init_db()
    await config_manager.initialize()
    await knowledge_base.initialize()
    if scheduler:
        try:
            scheduler.add_job(
                active_behavior_manager.run_tick,
                "interval",
                minutes=1,  # 改为每分钟触发一次，具体间隔由各群组配置决定
                id="virtual_friends_active_behavior",
                replace_existing=True,
                max_instances=1,
                coalesce=True,
            )
            scheduler.add_job(
                check_daily_summary,
                "cron",
                minute="*",
                id="virtual_friends_daily_summary",
                replace_existing=True
            )
            logger.success("定时任务已启动")
        except Exception as e:
            logger.error(f"启动定时任务失败: {e}")
            logger.info("主动行为调度已启动, 基础心跳 1 分钟")
        except Exception as exc:
            logger.error(f"主动行为调度启动失败: {exc}")

@driver.on_shutdown
async def shutdown():
    # 1. 关闭数据库连接
    from tortoise import Tortoise
    try:
        conn = Tortoise.get_connection("default")
        await conn.execute_query("PRAGMA wal_checkpoint(TRUNCATE);")
        logger.debug("WAL checkpoint completed")
    except Exception as exc:
        logger.warning(f"WAL checkpoint failed: {exc}")
    await Tortoise.close_connections()
    logger.debug("数据库连接已关闭")

    # 2. 尝试优雅关闭 htmlrender 浏览器 (消除 Ctrl+C 报错)
    try:
        import nonebot.plugin
        if nonebot.plugin.get_plugin("nonebot_plugin_htmlrender"):
            from nonebot_plugin_htmlrender import get_browser
            browser = await get_browser()
            if browser and browser.is_connected():
                await browser.close()
                logger.debug("HtmlRender 浏览器已手动关闭")
    except Exception:
        # 忽略所有浏览器关闭异常
        pass
