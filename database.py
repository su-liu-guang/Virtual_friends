from tortoise import fields, Tortoise
from tortoise.models import Model
from datetime import datetime

class ImageCache(Model):
    md5 = fields.CharField(max_length=32, pk=True)
    caption = fields.TextField()
    created_at = fields.DatetimeField(auto_now_add=True)
    
    class Meta: # type: ignore
        table = "image_cache"

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
    await Tortoise.generate_schemas()
