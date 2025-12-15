import json
import jinja2
import re
import time
import json_repair
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, List
from nonebot import logger, require

try:
    require("nonebot_plugin_htmlrender")
    from nonebot_plugin_htmlrender import html_to_pic
except Exception:
    logger.warning("未检测到 nonebot_plugin_htmlrender，无法生成图片总结")
    html_to_pic = None

from .database import Message
from .clients import ChatClient
from .config import ConfigManager

class DailySummaryGenerator:
    def __init__(self, chat_client: ChatClient, config_manager: ConfigManager):
        self.chat_client = chat_client
        self.config_manager = config_manager
        self.template_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(str(Path(__file__).parent)),
            enable_async=True
        )

    async def get_messages_by_date(self, group_id: str, target_date: datetime) -> List[Message]:
        # 保证存在时区信息，默认使用本地时区
        if target_date.tzinfo is None:
            target_date = target_date.replace(tzinfo=datetime.now().astimezone().tzinfo)
        
        # 以本地日界为准，再转换成 UTC 查询，避免时区导致跨天
        local_date = target_date.astimezone()
        start_local = local_date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_local = local_date.replace(hour=23, minute=59, second=59, microsecond=999999)
        start_of_day = start_local.astimezone(timezone.utc)
        end_of_day = end_local.astimezone(timezone.utc)
        
        messages = await Message.filter(
            group_id=group_id,
            timestamp__gte=start_of_day,
            timestamp__lte=end_of_day
        ).order_by("timestamp").all()
        
        return messages

    async def generate_report(self, group_id: str, target_date: Optional[datetime] = None) -> Optional[bytes]:
        if not html_to_pic:
            logger.error("缺少 nonebot_plugin_htmlrender 依赖")
            return None
            
        if target_date is None:
            target_date = datetime.now().astimezone()
            
        messages = await self.get_messages_by_date(group_id, target_date)
        
        # 1. 基础统计
        hour_counts = [0] * 24
        text_lines = []
        nickname_map = {} # 提前构建映射
        user_stats = {} # 统计用户数据: {uid: {'msg': 0, 'len': 0, 'img': 0}}
        
        config = self.config_manager.get_instance_config(group_id)
        
        # 获取 Bot ID 和 显示名称
        try:
            import nonebot
            bot = nonebot.get_bot()
            bot_id = str(bot.self_id)
            
            # 获取 Bot 在群内的名片或昵称
            try:
                member_info = await bot.get_group_member_info(group_id=int(group_id), user_id=int(bot_id))
                bot_name = member_info.get("card") or member_info.get("nickname") or config.get("persona_name", "AI助手")
            except Exception as e:
                logger.warning(f"获取Bot群名片失败: {e}")
                bot_name = config.get("persona_name", "AI助手")
        except Exception:
            bot_id = ""
            bot_name = config.get("persona_name", "AI助手")
        
        if messages:
            for msg in messages:
                # 转换为本地时间
                local_timestamp = msg.timestamp.astimezone()
                
                # 统计时间分布
                h = local_timestamp.hour
                if 0 <= h < 24:
                    hour_counts[h] += 1
                    
                # 确定显示名称 (uid)
                is_bot = str(msg.user_id) == bot_id
                if is_bot:
                    uid = bot_name
                else:
                    uid = msg.user_nickname or msg.user_id or "未知用户"
                
                # 更新映射 (确保 Bot 也能被映射到 QQ)
                if msg.user_id:
                    nickname_map[uid] = str(msg.user_id)
                
                # 统计发言数 (排除 Bot)
                if not is_bot:
                    if uid not in user_stats:
                        user_stats[uid] = {'msg': 0, 'len': 0, 'img': 0}
                    
                    stats = user_stats[uid]
                    stats['msg'] += 1
                    stats['len'] += len(msg.content) if msg.content else 0
                    if msg.image_md5:
                        stats['img'] += 1
                    
                # 文本记录 (用于 AI)
                if msg.content:
                    time_str = local_timestamp.strftime("%H:%M")
                    # 给 Bot 消息添加标签，方便 AI 识别并排除
                    display_uid = f"[AI助手]{uid}" if is_bot else uid
                    text_lines.append(f"[{time_str}] {display_uid}: {msg.content}")
        else:
            logger.info(f"群 {group_id} 今日无消息，将生成空数据报告")

        # 2. AI 生成结构化数据
        if text_lines:
            # 明确告知 AI 自己的身份
            bot_identity = f"你的名字是: {bot_name} (在聊天记录中显示为 [AI助手]{bot_name})"
            context = f"{bot_identity}\n\n" + "\n".join(text_lines)
        else:
            context = "(今日无消息记录，请编造一份有趣的虚构日报，假设群友们都在潜水或者发生了什么神秘事件)"
            
        t_start_ai = time.time()
        json_str = await self.chat_client.generate_daily_summary_data(context)
        t_end_ai = time.time()
        logger.info(f"[Performance] AI 生成耗时: {t_end_ai - t_start_ai:.2f}s")
        logger.debug(f"AI Summary Raw Response: {json_str}")
        
        data = None
        # 尝试解析，如果失败则让 AI 修复
        for attempt in range(2):
            try:
                # 清理可能的 markdown 标记
                clean_json_str = json_str
                if "```json" in clean_json_str:
                    clean_json_str = clean_json_str.split("```json")[1].split("```")[0]
                elif "```" in clean_json_str:
                    clean_json_str = clean_json_str.split("```")[1].split("```")[0]
                
                # 尝试解析
                try:
                    data = json.loads(clean_json_str)
                except Exception:
                    # 使用 json_repair 库修复
                    logger.warning("标准 JSON 解析失败，尝试使用 json_repair 修复...")
                    data = json_repair.loads(clean_json_str)
                
                # 有时候 AI 会返回双重编码的 JSON 字符串
                if isinstance(data, str):
                    try:
                        data = json.loads(data)
                    except Exception:
                        pass
                
                if not isinstance(data, dict):
                    raise ValueError(f"解析结果不是字典，而是 {type(data)}")
                    
                break # 解析成功，跳出循环
            except Exception as e:
                logger.warning(f"解析 AI 返回的 JSON 失败 (尝试 {attempt+1}/2): {e}")
                if attempt == 0:
                    logger.info("正在请求 AI 修复 JSON 格式...")
                    try:
                        fix_prompt = f"你生成的 JSON 数据解析失败，报错为：{e}。\n请修复以下 JSON 数据，直接输出修复后的 JSON 字符串，不要包含 markdown 标记，不要输出任何其他内容。\n\n{json_str}"
                        # 复用 chat_client 的 generate_response 方法，构造一个临时的 messages
                        from openai.types.chat import ChatCompletionMessageParam
                        messages = [{"role": "user", "content": fix_prompt}]
                        json_str = await self.chat_client.generate_response(messages, retry=1)
                        logger.debug(f"AI 修复后的 JSON: {json_str}")
                    except Exception as fix_e:
                        logger.error(f"请求 AI 修复 JSON 失败: {fix_e}")
                        return None
                else:
                    logger.error(f"最终解析失败。Raw: {json_str}")
                    return None
        
        if data is None:
            return None
            
        # 3. 补充/修正数据
        # config 已在上面获取
        data["date"] = target_date.strftime("%Y.%m.%d")
        data["group_name"] = config.get("group_name", group_id)
        
        # nickname_map 已在上面构建完成，无需重复构建

        # 辅助函数：获取 QQ 号
        def get_qq(name):
            # 去除可能的 [AI助手] 前缀
            clean_name = name.replace("[AI助手]", "").strip()
            
            # 0. 特殊处理 Bot
            if clean_name == bot_name or clean_name == "AI助手" or name == bot_name:
                return bot_id

            # 1. 查表
            if clean_name in nickname_map:
                return nickname_map[clean_name]
            # 2. 自身就是 QQ 号? (防止 AI 直接输出 QQ)
            if clean_name.isdigit() and len(clean_name) > 5:
                return clean_name
            return "0"

        # 为 users 补充 qq 号，并过滤掉机器人，同时注入真实统计数据
        valid_users = []
        for user in data.get("users", []):
            qq = get_qq(user["name"])
            # 过滤条件：QQ号是Bot ID，或者名字就是Bot ID，或者名字是Bot显示名，或者名字包含[AI助手]
            if str(qq) == bot_id or user["name"] == bot_id or user["name"] == bot_name or "[AI助手]" in user["name"]:
                continue
            user["qq"] = qq
            
            # 注入真实统计数据
            # 尝试通过名字匹配统计数据
            stats = user_stats.get(user["name"], {'msg': 0, 'len': 0, 'img': 0})
            # 如果名字没匹配上，尝试通过 QQ 反查 (可能 AI 输出的名字和 map 里的 key 有细微差别)
            if stats['msg'] == 0 and qq != "0":
                # 遍历 user_stats 的 key，看哪个 key 对应的 qq 是这个
                for u_name, u_stats in user_stats.items():
                    if get_qq(u_name) == qq:
                        stats = u_stats
                        break
            
            # 构造真实的 stats
            avg_len = int(stats['len'] / stats['msg']) if stats['msg'] > 0 else 0
            user["stats"] = {
                "msg": stats['msg'],
                "len": stats['len']
            }
            
            valid_users.append(user)
        data["users"] = valid_users
        # 为 quotes 补充 qq 号 (允许机器人)
        for quote in data.get("quotes", []):
            qq = get_qq(quote["user"])
            quote["qq"] = qq

        # 过滤 stats 中的机器人
        valid_stats = []
        for stat in data.get("stats", []):
            # 尝试通过 value (昵称) 反查 QQ
            qq = get_qq(stat["value"])
            if str(qq) == bot_id or stat["value"] == bot_id or stat["value"] == bot_name:
                continue
            valid_stats.append(stat)
        data["stats"] = valid_stats
            
        # 为 topics 补充 qq 号 (如果需要)
        for topic in data.get("topics", []):
            topic["users_qq"] = [get_qq(u) for u in topic.get("users", [])]
            
            # 处理 summary 中的 [Avatar:昵称] 标签
            summary_text = topic.get("summary", "")
            
            def replace_avatar(match):
                name = match.group(1)
                qq = get_qq(name)
                clean_name = name.replace("[AI助手]", "").strip()
                return f'<span class="inline-flex items-center bg-yellow-100 border border-black rounded px-1 mx-0.5 text-xs font-bold"><img class="w-4 h-4 rounded-full mr-1" src="http://q1.qlogo.cn/g?b=qq&nk={qq}&s=100">{clean_name}</span>'
            
            # 替换 [Avatar:xxx]
            summary_text = re.sub(r'\[Avatar:(.*?)\]', replace_avatar, summary_text)
            
            # 处理 (AI吐槽: ...) -> 变灰变小
            summary_text = re.sub(r'\((AI吐槽:.*?)\)', r'<span class="text-gray-400 text-[10px] block mt-1">\1</span>', summary_text)
            
            topic["summary"] = summary_text

        # 计算活跃度百分比 (用于图表)
        max_hour = max(hour_counts) if max(hour_counts) > 0 else 1
        # 生成 24 个高度百分比
        data["activity_trend"] = [int((c / max_hour) * 100) for c in hour_counts]
        
        # 添加静态资源路径
        data["static_path"] = (Path(__file__).parent / "static").as_uri()
        
        # 4. 渲染 HTML
        try:
            t_start_render = time.time()
            template = self.template_env.get_template("Daily_Report_Template.html")
            html = await template.render_async(**data)
            
            # 5. 生成图片
            # 宽度 520px，高度自适应
            img = await html_to_pic(
                html, 
                viewport={"width": 520, "height": 100}, 
                type="jpeg", 
                quality=90
            )
            t_end_render = time.time()
            logger.info(f"[Performance] 图片渲染耗时: {t_end_render - t_start_render:.2f}s")
            return img
        except Exception as e:
            logger.error(f"渲染 HTML 或生成图片失败: {e}")
            return None