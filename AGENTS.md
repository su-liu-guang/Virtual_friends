# Virtual Friends — AGENTS.md

## Overview

NoneBot2 插件，为 QQ 群聊提供 AI 驱动的虚拟好友。功能包括：人设对话、三层记忆整理、向量知识库检索、主动行为调度、每日 HTML 日报。

## Architecture

```
__init__.py         # 入口：命令注册、生命周期、消息处理主循环
active_behavior.py  # 主动行为：自主判断何时在群里发言
clients.py          # AI 服务封装：Vision/Chat/Embedding 三客户端
config.py           # 配置管理：personas.json + groups.json 读写
database.py         # Tortoise ORM 模型：Message/Summary/ImportantEvent/ImageCache
knowledge.py        # 向量知识库：Markdown 文档 → 分块 → 向量 → 语义检索
logic.py            # 上下文构建：system prompt 拼装、格式约束与清洗
scheduler.py        # 后台调度：消息批量处理、摘要归档、事实库维护
summary.py          # 日报生成：AI 结构化数据 + 本地统计 → HTML → JPEG
```

### Data Flow

```
用户消息 → __init__.py (handle_message)
  ├─ knowledge.py (向量检索 → system prompt)
  ├─ logic.py (ContextBuilder: 拼装 persona + 摘要 + 事实 + 最近消息)
  ├─ clients.py (ChatClient 生成回复, VisionClient 识别图片)
  └─ scheduler.py (后台异步入队: 批量摘要 + 事实提取)
       └─ active_behavior.py (定时主动发言)
```

### Memory Tier System

| 层级 | 触发条件 | 说明 |
|------|---------|------|
| L1 | 每 50 条消息 | 详细摘要 |
| L2 | 每 80 条 L1 | 叙事概括 |
| L3 | 每 30 条 L2 | 宏观印象 |

所有未归档的 L1/L2/L3 摘要和有效事实**全量注入** system prompt（不经过向量检索）。

## Key Patterns

### 格式约束（强制 `<persona_reply>` 标签）

- `logic.py` 输出 `OUTPUT_FORMAT_PROMPT`（内容在 `logic.py:14`），在所有 system prompt 最前面强制模型输出 `<persona_reply>...</persona_reply>`
- `generate_with_format_retry()` (`logic.py:133`)：首次失败时用 `FORMAT_RETRY_SYSTEM` 重试一次
- `sanitize_persona_reply()` (`logic.py:45`)：从标签中提取纯文本，移除元信息行

### JSON 格式提取

- `extract_facts_v2()` (`clients.py:196`)：结构化事实提取，system prompt 强制输出 JSON 数组
- `extract_maintenance_actions()` (`clients.py:275`)：事实库维护，输出 `{expire, conflicts, merges}`
- `generate_daily_summary_data()` (`clients.py:328`)：日报数据，优先 `json_mode`，回退普通模式
- 以上均有 JSON 解析失败后的重试循环（发送修复 prompt 再次调用）

### 重试与容错

- 所有 API 调用使用指数退避 (`asyncio.sleep(2**attempt)`)
- `json_mode` 回退到 `disable_thinking` 普通模式
- `json_repair` 库兜底解析损坏 JSON

## Database

SQLite (aiosqlite, WAL mode)，路径：`data/Virtual_friends/database.sqlite`

| 表 | 关键字段 | 用途 |
|----|---------|------|
| `messages` | group_id, role(user/ai), content, image_md5, is_processed | 聊天记录 |
| `summaries` | group_id, level(1/2/3), content, time_range, is_archived | 三层摘要 |
| `important_events` | group_id, event_content, fact_type, confidence, validity, expires_at | 结构化事实 |
| `image_cache` | md5(PK), caption | 图片识别缓存 |

初始化：`init_db()` at `database.py`

## Configuration

### 文件存储
- `data/Virtual_friends/personas.json` — 人设定义
- `data/Virtual_friends/groups.json` — 每群配置（persona 绑定、回复率、主动模式参数等）
- `data/Virtual_friends/knowledge/` — Markdown 知识文档

### 环境变量（通过 `.env.dev`）
- `chat_api_key/base_url/model_name` — 文本模型
- `vision_api_key/base_url/model_name` — 视觉模型
- `embedding_api_key/base_url/model_name` — 向量模型
- `reranker_api_key/url/model` — Reranker（可选）

### 配置项缩写映射
`logic.py` 中的 `KEY_ALIAS` / `ALIAS_TO_KEY` 提供中文↔英文键名转换

## Development Notes

### 依赖注入方式
- `__init__.py` 实例化所有全局对象并传入构造函数
- `clients.py` 例外：每个客户端内部创建自己的 `ConfigManager()`（非 DI）

### Prompt 存放位置
- 格式约束 prompt：`logic.py` 顶部常量
- LLM 功能 prompt：内联在各 `clients.py` 方法中

### 特殊语法
- `<persona_reply>...</persona_reply>` — AI 回复必须包裹在此标签内
- `{{发送图片:xxx}}` — 触发图片发送宏
- `[Avatar:昵称]` — 日报中嵌入 QQ 头像 HTML
- `[AI助手]名称` — 日报中标记 bot 自身消息以排除

### 指令权限
- `/人生启动` `/世界终结` `/重载配置` / 日报类 — 仅超管
- `/vf帮助` `/记忆状态` / 人设查看 — 所有用户

### 当前已知问题
- `clients.py` 中每个客户端创建独立 `ConfigManager` 实例（而非注入），可能造成不一致但实际无害

## Commands

```bash
# 运行（需在 nonebot 项目根目录）
nb run

# 依赖安装
pip install -r requirements.txt
```

## File Map

```
__init__.py        770L  入口、18 个命令、定时任务注册
active_behavior.py 267L  ActiveBehaviorManager
clients.py         430L  VisionClient / ChatClient / EmbeddingClient
config.py          261L  ConfigManager
database.py         66L  4 models + init_db()
knowledge.py       317L  KnowledgeBase + RerankerClient
logic.py           302L  ContextBuilder + format helpers
scheduler.py       241L  MemoryScheduler
summary.py         311L  DailySummaryGenerator
Daily_Report_Template.html 275L  Jinja2 日报模板
static/                   tailwindcss.js, lucide.js
requirements.txt          nonebot2, tortoise-orm, openai, aiohttp, aiosqlite, htmlrender
```
