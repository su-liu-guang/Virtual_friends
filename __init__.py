from nonebot import on_message, on_command, logger, require
from nonebot.adapters.onebot.v11 import Bot, MessageEvent, GroupMessageEvent, Message as OB11Message
from nonebot.adapters.onebot.v11.permission import GROUP_OWNER
from nonebot.permission import SUPERUSER
from nonebot.exception import MatcherException
from nonebot.params import CommandArg
import json
from datetime import datetime
import asyncio
import random

from .config import ConfigManager
from .database import init_db, Message, Summary, ImportantEvent
from .clients import VisionClient, ChatClient
from .logic import ContextBuilder, process_image_message
from .scheduler import MemoryScheduler
from .active_behavior import ActiveBehaviorManager

# 初始化
config_manager = ConfigManager()
vision_client = VisionClient()
chat_client = ChatClient()
context_builder = ContextBuilder(config_manager)
memory_scheduler = MemoryScheduler(chat_client)
active_behavior_manager = ActiveBehaviorManager(config_manager, chat_client, context_builder, memory_scheduler)

try:
    scheduler = require("nonebot_plugin_apscheduler").scheduler
except RuntimeError:
    scheduler = None
    logger.warning("未安装 nonebot_plugin_apscheduler, 主动行为功能已禁用")

# 工具函数
def get_group_id(event: MessageEvent) -> str:
    return str(event.group_id) if isinstance(event, GroupMessageEvent) else str(event.user_id)


WEEKDAY_CN = ["一", "二", "三", "四", "五", "六", "日"]


def now_with_minute_precision() -> datetime:
    return datetime.now().replace(second=0, microsecond=0)


def get_weekday_label(dt: datetime) -> str:
    return f"星期{WEEKDAY_CN[dt.isoweekday() - 1]}"

# 消息处理器
message_handler = on_message(priority=10, block=False)

@message_handler.handle()
async def handle_message(bot: Bot, event: MessageEvent):
    # 过滤机器人自己发送的消息
    if str(event.user_id) == str(bot.self_id):
        return
    
    group_id = get_group_id(event)
    
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

    text_content = event.get_plaintext().strip()
    logger.info(f"收到消息 [群组: {group_id}] [用户: {event.user_id}]: {text_content[:50]}...")
    
    image_md5 = None
    if has_image:
        for seg in event.message:
            if seg.type == "image":
                image_url = seg.data["url"]
                summary = seg.data.get("summary", "")
                is_sticker = summary == "[动画表情]"
                logger.info(f"检测到图片: {image_url[:50]}... (Summary: {summary})")
                image_md5 = await process_image_message(image_url, vision_client, is_sticker=is_sticker)
                logger.success(f"图片处理完成, MD5: {image_md5}")
                break
    
    timestamp = now_with_minute_precision()
    weekday_label = get_weekday_label(timestamp)
    await Message.create(
        group_id=group_id,
        role="user",
        content=text_content,
        image_md5=image_md5,
        timestamp=timestamp,
        weekday=weekday_label,
        is_processed=False
    )
    

    
    if not should_reply:
        asyncio.create_task(memory_scheduler.check_and_process(group_id))
        return
    
    # 构建上下文并生成回复
    context = await context_builder.build_context(
        group_id=group_id,
        user_nickname=event.sender.card or event.sender.nickname or "用户",
        current_time=timestamp
    )
    response = await chat_client.generate_response(context)
    response = response.strip()  # 去除首尾空格和换行
    
    # 存储并发送回复
    reply_timestamp = now_with_minute_precision()
    await Message.create(
        group_id=group_id,
        role="ai",
        content=response,
        timestamp=reply_timestamp,
        weekday=get_weekday_label(reply_timestamp),
        is_processed=False
    )
    await message_handler.send(response)
    
    # 触发后台任务
    asyncio.create_task(memory_scheduler.check_and_process(group_id))

# 指令：切换人设
switch_persona = on_command("切换人设", aliases={"切换提示词"}, priority=5, block=True)

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
    await switch_persona.finish(f"已切换至人设: {personas[arg_text]['description']}")

# 指令：记忆状态
memory_status = on_command("记忆状态",aliases={"查看记忆"}, priority=5, block=True)

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

# 指令：遗忘
forget_cmd = on_command("遗忘", priority=5, block=True, permission=SUPERUSER | GROUP_OWNER)

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

# 指令：清空记忆
clear_memory = on_command("清空记忆", aliases={"重置记忆"}, priority=5, block=True, permission=SUPERUSER | GROUP_OWNER)

@clear_memory.handle()
async def handle_clear_memory(bot: Bot, event: MessageEvent):
    group_id = get_group_id(event)
    try:
        await Message.filter(group_id=group_id).delete()
        await Summary.filter(group_id=group_id).delete()
        await ImportantEvent.filter(group_id=group_id).delete()
        await clear_memory.finish("✅ 已清空当前群组的所有记忆")
    except MatcherException:
        raise
    except Exception as e:
        await clear_memory.finish(f"❌ 清空失败: {e}")

# 指令：提示词列表
persona_list = on_command("提示词列表", aliases={"人设列表"}, permission=SUPERUSER | GROUP_OWNER, priority=5, block=True)

@persona_list.handle()
async def handle_persona_list(bot: Bot, event: MessageEvent):
    personas = config_manager.get_personas()
    msg = "\n".join([f"• {k}: {v.get('description', '无')}" for k, v in personas.items()])
    await persona_list.finish(f"📝 可用人设:\n{msg}\n使用 /切换人设 [名称] 切换")

# 指令：查看提示词
view_persona = on_command("查看提示词", aliases={"查看人设"}, priority=5, block=True)

@view_persona.handle()
async def handle_view_persona(bot: Bot, event: MessageEvent, args: OB11Message = CommandArg()):
    arg_text = args.extract_plain_text().strip()
    group_id = get_group_id(event)
    
    if not arg_text:
        arg_text = config_manager.get_instance_config(group_id).get("persona_name", "default")
    
    persona = config_manager.get_personas().get(arg_text)
    if not persona:
        await view_persona.finish(f"人设 '{arg_text}' 不存在")
        
    await view_persona.finish(f"📋 {arg_text} ({persona.get('description')})\n\n{persona.get('prompt')}")

# 指令：添加提示词
add_persona = on_command("添加提示词", aliases={"添加人设","增加提示词","增加人设"}, priority=5, block=True, permission=SUPERUSER | GROUP_OWNER)

@add_persona.handle()
async def handle_add_persona(bot: Bot, event: MessageEvent, args: OB11Message = CommandArg()):
    text = args.extract_plain_text().strip()
    parts = text.split(maxsplit=2)
    
    if len(parts) < 3:
        await add_persona.finish("用法: /添加提示词 名称 描述 提示词")
    
    name, desc, prompt = parts[0], parts[1], parts[2]
    if config_manager.add_persona(name, prompt, desc):
        await add_persona.finish(f"✅ 已添加人设 '{name}'")
    else:
        await add_persona.finish("❌ 添加失败")

# 指令：删除提示词
delete_persona = on_command("删除提示词", aliases={"删除人设"}, priority=5, block=True, permission=SUPERUSER)

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

# 指令：人生启动
enable_plugin = on_command("人生启动", aliases={"世界开启","故事开始"},priority=5, block=True, permission=SUPERUSER | GROUP_OWNER)

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

# 指令：世界终结
disable_plugin = on_command("世界终结", priority=5, block=True, permission=SUPERUSER | GROUP_OWNER)

@disable_plugin.handle()
async def handle_disable_plugin(bot: Bot, event: GroupMessageEvent):
    group_id = str(event.group_id)
    if config_manager.remove_from_whitelist(group_id):
        await disable_plugin.finish("🌙 世界终结... 本群已禁用插件")
    else:
        await disable_plugin.finish("⚠️ 本群未启用")

# 指令：查看白名单
view_whitelist = on_command("查看白名单", priority=5, block=True, permission=SUPERUSER | GROUP_OWNER)

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

# 指令：查看配置
view_config = on_command("查看配置", aliases={"vf配置", "当前配置"}, priority=5, block=True, permission=SUPERUSER | GROUP_OWNER)

@view_config.handle()
async def handle_view_config(bot: Bot, event: MessageEvent):
    group_id = get_group_id(event)
    if not config_manager.is_in_whitelist(group_id):
        await view_config.finish("⚠️ 本群未启用插件")
    
    config = config_manager.get_instance_config(group_id)
    # 复制一份配置，避免修改原对象
    display_config = config.copy()
    
    # 移除不建议手动修改或不应发送的字段
    for k in ("whitelisted", "group_name"):
        display_config.pop(k, None)
    
    await view_config.finish(f"⚙️ 当前配置 (复制修改后使用 /修改配置 发送):\n{json.dumps(display_config, ensure_ascii=False, indent=2)}")

# 指令：修改配置
update_config = on_command("修改配置", aliases={"vf设置", "更新配置"}, priority=5, block=True, permission=SUPERUSER | GROUP_OWNER)

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
        
    # 安全过滤：只允许修改特定的配置项
    safe_config = {}
    valid_keys = [
        "persona_name", "reply_rate", "active_mode", "active_hours", 
        "active_check_interval", "idle_trigger_probability", "silence_threshold",
        "group_name" # 允许修改群名备注
    ]
    
    for k, v in new_config.items():
        if k in valid_keys:
            safe_config[k] = v
            
    if not safe_config:
        await update_config.finish("⚠️ 未检测到有效的配置项")
        
    config_manager.update_instance_config(group_id, safe_config)
    
    # 反馈更新后的配置
    final_config = config_manager.get_instance_config(group_id)
    # 同样移除 whitelisted
    display_final = final_config.copy()
    if "whitelisted" in display_final:
        del display_final["whitelisted"]
        
    await update_config.finish(f"✅ 配置已更新:\n{json.dumps(display_final, ensure_ascii=False, indent=2)}")

# 指令：帮助
help_cmd = on_command("vf帮助", aliases={"vf菜单", "vf指令列表"}, priority=5, block=True)

@help_cmd.handle()
async def handle_help(bot: Bot, event: MessageEvent):
    await help_cmd.finish("""🤖 Virtual Friends 指令列表
━━━━━━━━━━━━━━
基础指令:
• /提示词列表 - 查看所有可用人设
• /查看提示词 [名称] - 查看指定人设详情
• /记忆状态 - 查看当前群聊的记忆统计

管理指令 (超管/群主)
• /切换人设 [名称] - 切换当前群聊的 AI 人设
• /人生启动 - 在当前群启用插件 (加入白名单)
• /世界终结 - 在当前群禁用插件 (移出白名单)
• /清空记忆 - 删除当前群的所有记忆数据
• /遗忘 [关键词] - 删除包含关键词的特定记忆
• /添加提示词 <名称> <描述> <内容> - 添加新人设
• /删除提示词 [名称] - 删除指定人设 (仅超管)
• /查看白名单 - 查看已启用插件的群
• /查看配置 - 查看当前群组的详细配置
• /修改配置 <JSON> - 修改当前群组配置""")


# 启动初始化
import nonebot
driver = nonebot.get_driver()

@driver.on_startup
async def startup():
    await init_db()
    await config_manager.initialize()
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
            logger.info("主动行为调度已启动, 基础心跳 1 分钟")
        except Exception as exc:
            logger.error(f"主动行为调度启动失败: {exc}")

@driver.on_shutdown
async def shutdown():
    from tortoise import Tortoise
    await Tortoise.close_connections()
    logger.debug("数据库连接已关闭")
