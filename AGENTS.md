# Virtual Friends — AGENTS.md

## Overview

NoneBot2 插件，为 QQ 群聊提供 AI 驱动的虚拟好友。功能包括：人设对话、三层记忆整理、向量知识库检索、主动行为调度、每日 HTML 日报。

## Architecture

```
__init__.py         # 入口：命令注册、生命周期、消息处理主循环
active_behavior.py  # 主动行为：自主判断何时在群里发言
clients.py          # AI 服务封装：Chat/Embedding 客户端
cache_metrics.py    # DeepSeek prompt cache 用量 JSONL 监控
config.py           # 配置管理：personas.json + groups.json 读写
database.py         # Tortoise ORM 模型：消息、摘要、永久图片文件与图片组映射
image_upload.py     # Files API 持久化后台上传、退避重试与重启恢复
knowledge.py        # 向量知识库：Markdown 文档 → 分块 → 向量 → 语义检索
logic.py            # 上下文构建：system prompt 拼装、回复清洗与图片处理
scheduler.py        # 后台调度：消息批量处理、摘要归档、事实库维护
summary.py          # 日报生成：AI 结构化数据 + 本地统计 → HTML → JPEG
```

### Data Flow

```
用户消息 → __init__.py (handle_message)
  ├─ logic.py (图片缓存命中用 file_id，未命中首次 Base64 内联)
  ├─ image_upload.py (本地落盘 → 后台 Files API → 永久缓存)
  ├─ knowledge.py (向量检索 → 当前请求的独立临时消息)
  ├─ logic.py (ContextBuilder: 拼装 persona + 摘要 + 未处理消息)
  ├─ clients.py (ChatClient 统一生成文本与图像理解回复)
  └─ scheduler.py (后台异步入队: 批量摘要 + 事实提取)
       └─ active_behavior.py (定时主动发言)
```

### Memory Tier System

| 层级 | 触发条件 | 说明 |
|------|---------|------|
| L1 | 每 50 条消息 | 详细摘要 |
| L2 | 每 20 条 L1 | 叙事概括 |
| L3 | 每 10 条 L2 | 宏观印象 |

未归档 L1/L2 全量注入 system prompt；L3 注入最新 5 条。当前摘要周期的未处理消息按时间追加，不使用滑动窗口。

## Key Patterns

### 回复生成与清洗

- 普通聊天和主动发言均只执行一轮回复生成；API 请求异常仍沿用客户端重试策略
- 不要求模型输出 XML 标签，也不会因为回复格式触发额外模型调用
- `sanitize_reply()` 仅移除代码块、时间戳和常见元信息行

### JSON 格式提取

- `extract_facts_v2()` (`clients.py:196`)：结构化事实提取，system prompt 强制输出 JSON 数组
- `extract_maintenance_actions()` (`clients.py:275`)：事实库维护，输出 `{expire, conflicts, merges}`
- `generate_daily_summary_data()` (`clients.py:328`)：日报数据，优先 `json_mode`，回退普通模式
- 以上均有 JSON 解析失败后的重试循环（发送修复 prompt 再次调用）

### 重试与容错

- 所有 API 调用使用指数退避 (`asyncio.sleep(2**attempt)`)
- 普通图片首次以内联方式调用视觉模型，不同步等待 Files API
- Files API 任务先写入 SQLite 与 `pending_images/`，后台单并发指数退避，重启后恢复
- 内联原图总量上限 30 MiB；超过预算的大图才同步使用 Files API
- L1/L2/L3 摘要前 4 次必须少于 1000 字，第 5 次放宽到最多 1500 字，第 6 次不限制字数但仍须非空；最多总计 6 次 API 请求
- 摘要连续 6 次失败时不写入、不归档，也不机械截断
- `json_mode` 回退到 `disable_thinking` 普通模式
- `json_repair` 库兜底解析损坏 JSON

## Database

SQLite (aiosqlite, WAL mode)，路径：`data/Virtual_friends/memory.db`

| 表 | 关键字段 | 用途 |
|----|---------|------|
| `messages` | group_id, role, content, api_content, image_md5, is_processed | 清洗正文与 API 原始回复 |
| `summaries` | group_id, level(1/2/3), content, time_range, is_archived | 三层摘要 |
| `important_events` | group_id, event_content, fact_type, confidence, validity, expires_at | 结构化事实 |
| `image_file_cache` | scoped_md5(PK), file_id, filename | 按 API Key 隔离的 DeepSeek Files API 永久文件引用 |
| `image_batch_cache` | md5(PK), api_scope, files | 消息图片组与有序 file_id/大小映射 |
| `pending_image_uploads` | scoped_md5(PK), file_path, attempts, next_retry_at | 可恢复的单图上传任务 |
| `pending_image_batches` | md5(PK), api_scope, images | 等待全部 file_id 就绪的有序图片组 |

初始化：`init_db()` at `database.py`

## Configuration

### 文件存储
- `data/Virtual_friends/personas.json` — 人设定义
- `data/Virtual_friends/groups.json` — 每群配置（persona 绑定、回复率、主动模式参数等）
- `data/Virtual_friends/knowledge/` — Markdown 知识文档
- 知识库中的文件或子目录名以 `_` 开头时暂不加载；移除前缀后会重新加入索引
- `data/Virtual_friends/logs/cache_metrics.jsonl` — 缓存 token JSONL，10 MiB 轮转并保留 30 天
- `data/Virtual_friends/pending_images/` — Files API 成功前保留的原图，任务完成后自动清理

### 环境变量（通过 `.env.prod`）
- `chat_api_key/base_url/model_name` — 文本模型
- `chat_model_name=deepseek-v4-flash-vision-exp` — 统一的文本+视觉模型
- `vision_api_*` — 已停止使用，可暂时保留用于回滚
- `embedding_api_key/base_url/model_name` — 向量模型
- `reranker_api_key/url/model` — Reranker（可选）

### 配置项缩写映射
`logic.py` 中的 `KEY_ALIAS` / `ALIAS_TO_KEY` 提供中文↔英文键名转换

## Development Notes

### 依赖注入方式
- `__init__.py` 实例化所有全局对象并传入构造函数
- `clients.py` 例外：每个客户端内部创建自己的 `ConfigManager()`（非 DI）

### Prompt 存放位置
- LLM 功能 prompt：内联在各 `clients.py` 方法中

### 特殊语法
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

# 缓存与摘要模拟测试
python tests/test_cache_summary.py
```

## File Map

```
__init__.py        770L  入口、18 个命令、定时任务注册
active_behavior.py 267L  ActiveBehaviorManager
clients.py         ChatClient / EmbeddingClient
config.py          261L  ConfigManager
database.py         6 models + init_db()
image_upload.py     ImageUploadManager + 持久化重试队列
knowledge.py       317L  KnowledgeBase + RerankerClient
logic.py           302L  ContextBuilder + format helpers
scheduler.py       241L  MemoryScheduler
summary.py         311L  DailySummaryGenerator
Daily_Report_Template.html 275L  Jinja2 日报模板
static/                   tailwindcss.js, lucide.js
requirements.txt          nonebot2, tortoise-orm, openai, aiohttp, aiosqlite, htmlrender
```
