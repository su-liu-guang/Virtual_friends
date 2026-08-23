from tortoise import fields, Tortoise
from tortoise.models import Model


class ImageFileCache(Model):
    """DeepSeek Files API 永久文件引用缓存。"""

    # API 地址/Key 指纹与原图 MD5 的组合哈希，避免换 Key 后复用无权限的 file_id。
    md5 = fields.CharField(max_length=32, pk=True)
    file_id = fields.CharField(max_length=128)
    filename = fields.CharField(max_length=512)
    created_at = fields.DatetimeField(auto_now_add=True)

    class Meta: # type: ignore
        table = "image_file_cache"


class ImageBatchCache(Model):
    """消息图片组与有序 file_id 的映射。"""

    md5 = fields.CharField(max_length=32, pk=True)
    api_scope = fields.CharField(max_length=64)
    files = fields.JSONField()
    created_at = fields.DatetimeField(auto_now_add=True)
    updated_at = fields.DatetimeField(auto_now=True)

    class Meta: # type: ignore
        table = "image_batch_cache"


class Message(Model):
    id = fields.IntField(pk=True)
    group_id = fields.CharField(max_length=50, index=True)
    role = fields.CharField(max_length=10)  # user or ai
    content = fields.TextField()
    user_nickname = fields.CharField(max_length=64, null=True, description="消息发送者的昵称")
    user_id = fields.CharField(max_length=32, null=True, description="发送者 QQ 号或 Bot ID")
    image_md5 = fields.CharField(max_length=32, null=True)
    timestamp = fields.DatetimeField(index=True)
    display_time = fields.CharField(max_length=30, null=True, description="人类可读的时间格式 YYYY-MM-DD HH:MM")
    weekday = fields.CharField(max_length=10, null=True)
    is_processed = fields.BooleanField(default=False, index=True)
    reasoning_content = fields.TextField(null=True, description="DeepSeek 思考模式的思维链内容")
    api_content = fields.TextField(null=True, description="模型原始回复，仅用于 API 历史回放")
    
    class Meta: # type: ignore
        table = "messages"
        ordering = ["-timestamp"]

class Summary(Model):
    id = fields.IntField(pk=True)
    group_id = fields.CharField(max_length=50, index=True)
    level = fields.IntField(index=True)  # 1=细节, 2=叙事, 3=宏观
    content = fields.TextField()
    time_range = fields.CharField(max_length=100)
    is_archived = fields.BooleanField(default=False, index=True)
    created_at = fields.DatetimeField(auto_now_add=True)
    
    class Meta: # type: ignore
        table = "summaries"


async def init_db():
    """初始化数据库"""
    from pathlib import Path
    
    # 确保数据目录存在
    Path("data/Virtual_friends").mkdir(parents=True, exist_ok=True)
    
    await Tortoise.init(
        db_url='sqlite://data/Virtual_friends/memory.db',
        modules={'models': [__name__]}
    )
    conn = Tortoise.get_connection("default")

    # generate_schemas 不会为已有 SQLite 表增加字段，需做幂等兼容迁移。
    message_columns = await conn.execute_query_dict("PRAGMA table_info(messages)")
    if message_columns and not any(row.get("name") == "api_content" for row in message_columns):
        await conn.execute_query("ALTER TABLE messages ADD COLUMN api_content TEXT")

    # 图片摘要已彻底废弃；仅清理旧图片相关表，不影响消息和三层记忆。
    await conn.execute_query("DROP TABLE IF EXISTS image_cache")

    # 旧版 file_id 带 expires_at，不能代表永久文件，直接丢弃并重建。
    columns = await conn.execute_query_dict("PRAGMA table_info(image_file_cache)")
    if any(row.get("name") == "expires_at" for row in columns):
        await conn.execute_query("DROP TABLE IF EXISTS image_batch_cache")
        await conn.execute_query("DROP TABLE IF EXISTS image_file_cache")

    await Tortoise.generate_schemas()
