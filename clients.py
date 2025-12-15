import asyncio
from typing import List, Sequence
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam
from nonebot import logger
from .config import ConfigManager

class VisionClient:
    """视觉模型客户端 - 专注图像转文本"""
    
    def __init__(self):
        config = ConfigManager()
        self.api_key = config.get_env("vision_api_key")
        self.base_url = config.get_env("vision_api_url")
        self.model = config.get_env("vision_model_name", "gpt-4o-mini")
        
        
        if not self.api_key or not self.base_url:
            logger.error("Vision Client 配置不完整，请检查 .env.dev 文件中的 vision_api_key 和 vision_api_url")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key or "dummy",
            base_url=self.base_url or "https://api.openai.com/v1"
        )
    
    async def recognize_image(self, image_url: str, retry: int = 3, is_sticker: bool = False) -> str:
        """识别图片内容"""
        logger.debug(f"[Vision] 开始识别图片: {image_url[:50]}... (表情包: {is_sticker})")
        
        prompt = "请详细描述这张图片的内容,包括场景、物体、文字、情绪等。用简洁的中文回答。"
        if is_sticker:
            prompt = "这是一张动画表情包。请描述它的画面内容、文字(如果有)以及表达的情绪。用简洁的中文回答。"
        
        for attempt in range(retry):
            try:
                logger.debug(f"[Vision] 尝试 {attempt + 1}/{retry}")
                logger.debug(f"[Vision] 调用参数: model={self.model}, max_tokens=300")
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": prompt
                            },
                            {
                                "type": "image_url",
                                "image_url": {"url": image_url}
                            }
                        ]
                    }],
                    max_tokens=300
                )
                
                result = response.choices[0].message.content or "[图片识别无结果]"
                
                if is_sticker:
                    result = f"[动画表情] {result}"
                
                logger.debug(f"[Vision] 识别成功: {result[:100]}...")
                logger.debug(f"[Vision] Token 使用: prompt={response.usage.prompt_tokens if response.usage else 'N/A'}, completion={response.usage.completion_tokens if response.usage else 'N/A'}")
                return result
            
            except Exception as e:
                logger.error(f"[Vision] 识别失败 (尝试 {attempt + 1}/{retry}): {type(e).__name__}: {str(e)}")
                if attempt == retry - 1:
                    return f"[图片识别失败: {str(e)}]"
                await asyncio.sleep(2 ** attempt)
        
        return "[图片识别失败]"

class ChatClient:
    """聊天模型客户端 - 专注理解与生成"""
    
    def __init__(self):
        config = ConfigManager()
        self.api_key = config.get_env("chat_api_key")
        self.base_url = config.get_env("chat_api_url")
        self.model = config.get_env("chat_model_name", "deepseek-chat")
        
        if not self.api_key or not self.base_url:
            logger.error("Chat Client 配置不完整，请检查 .env.dev 文件中的 chat_api_key 和 chat_api_url")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key or "dummy",
            base_url=self.base_url or "https://api.openai.com/v1"
        )
    
    async def generate_response(self, messages: Sequence[ChatCompletionMessageParam], retry: int = 3) -> str:
        """生成对话回复"""
        
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.8,
            "extra_body":{"enable_thinking": True}
        }
        
        for attempt in range(retry):
            try:
                logger.debug(f"[Chat] 尝试 {attempt + 1}/{retry}")
                logger.debug(f"[Chat] 调用参数: model={self.model}, temperature=0.8")
                
                response = await self.client.chat.completions.create(**kwargs)
                
                result = response.choices[0].message.content or "..."
                logger.success(f"[Chat] 生成成功: {result[:100]}...")
                if response.usage:
                    prompt_tokens = response.usage.prompt_tokens
                    completion_tokens = response.usage.completion_tokens
                    total_tokens = response.usage.total_tokens
                    logger.debug(
                        f"[Chat] Token 使用: total={total_tokens}, prompt={prompt_tokens}, completion={completion_tokens}"
                    )
                else:
                    logger.debug("[Chat] Token 使用: N/A")
                return result
            
            except Exception as e:
                logger.error(f"[Chat] 生成失败 (尝试 {attempt + 1}/{retry}): {type(e).__name__}: {str(e)}")
                if attempt == retry - 1:
                    return "抱歉,我现在有点累了,稍后再聊吧..."
                await asyncio.sleep(2 ** attempt)
        
        return "抱歉,我现在有点累了,稍后再聊吧..."
    
    async def generate_summary(self, context: str, retry: int = 3) -> str:
        """生成总结"""
        logger.info(f"[Chat] 开始生成总结, 上下文长度: {len(context)}")
        messages: List[ChatCompletionMessageParam] = [{
            "role": "user",
            "content": (
                "总结以下群聊对话，忽略寒暄和无关细节，保留主要事件和讨论点。"
                "每条总结务必包含：1) 触发时间（可用消息时间或大致时间段）；"
                "2) 相关发送人昵称（ context 中的前缀已经包含昵称，请沿用）；"
                "3) 简要内容。"
                "以列表形式输出，确保可追溯到是谁在什么时候做了什么。\n\n"
                f"{context}"
            )
        }]
        return await self.generate_response(messages, retry)
    
    async def extract_facts(self, context: str, retry: int = 3) -> List[str]:
        """提取重要事实"""
        logger.info(f"[Chat] 开始提取事实, 上下文长度: {len(context)}")
        messages: List[ChatCompletionMessageParam] = [{
            "role": "user",
            "content": (
                "分析对话并提取长期记忆。请严格遵守以下过滤漏斗："
                "直接丢弃： 闲聊、情绪宣泄、模糊的打算（如“我想去...”）、对当下的评论。"
                "仅保留： 确定的行动计划（时间/地点）、永久性的人物属性（生日/过敏/职业）、对他人的郑重承诺。"
                "绝大多数情况下你应该返回“无”。"
                "仅当必定需要记录时，输出格式：[时间][昵称] ...（限2条）"
                "在输出前，请自问：'这条信息值得在一个月后被重新提起吗？' 如果不是，请输出无。\n\n"
                f"{context}"
            )
        }]
        
        result = await self.generate_response(messages, retry)
        if result.strip() == "无":
            logger.info("[Chat] 未提取到任何事实")
            return []
        
        facts = [line.strip() for line in result.split("\n") if line.strip() and not line.startswith("#")]
        logger.success(f"[Chat] 提取到 {len(facts)} 条事实")
        return facts

    async def generate_daily_summary_data(self, context: str, retry: int = 3) -> str:
        """生成每日总结的结构化数据 (JSON)"""
        logger.info(f"[Chat] 开始生成每日总结数据, 上下文长度: {len(context)}")
        
        prompt = """
你是一个群聊数据分析师。请根据提供的群聊记录，生成一份用于展示的 JSON 数据。
请严格按照以下 JSON 格式输出，不要包含 markdown 代码块标记，直接输出 JSON 字符串。

{
  "stats": [
    {"label": "称号1", "value": "用户A", "desc": "描述文本", "color": "red"},
    {"label": "称号2", "value": "用户B", "desc": "描述文本", "color": "green"},
    {"label": "称号3", "value": "用户C", "desc": "描述文本", "color": "yellow"},
    {"label": "称号4", "value": "统计值", "desc": "描述文本", "color": "purple"}
  ],
  "topics": [
    {"title": "话题标题", "hot": 5, "summary": "详细描述...", "users": ["参与者1", "参与者2"]},
    {"title": "话题标题", "hot": 4, "summary": "详细描述...", "users": ["参与者3"]}
  ],
  "users": [
    {
      "name": "用户昵称", 
      "title": "RPG职业/称号", 
      "stats": {"发言数": 0}, 
      "desc": "一句话角色描述"
    },
    {
      "name": "用户昵称", 
      "title": "RPG职业/称号", 
      "stats": {"发言数": 0}, 
      "desc": "一句话角色描述"
    }
  ],
  "quotes": [
    {"user": "用户昵称", "content": "金句内容..."},
    {"user": "用户昵称", "content": "金句内容..."}
  ],
  "fortune": {
    "luck": "大吉/中吉/小吉/凶...", 
    "text": "宜xxx，忌xxx (简短有趣)"
  }
}

要求：
1.数据清洗与 Stats：生成4个有趣的统计维度（如：龙王、复读机、熬夜冠军、开心果等），color 可选 red, green, yellow, purple, blue。请严格排除以下干扰数据：带有 [AI助手] 标签的用户、系统消息（如撤回、入群、文件上传）、以及纯表情包刷屏。
2.Topics：提取 3-5 个主要讨论话题。hot 字段必须是 1 到 5 之间的整数（用于前端渲染进度条，不可输出小数）。 summary 字段请提供详细的叙述，包含起因、经过、结果，具体描述“谁说了什么”。在提到群友昵称时，请务必使用 [Avatar:昵称] 的格式。注意：如果昵称带有 [AI助手] 前缀（例如 [AI助手]妖精爱莉），在 [Avatar:...] 标签中必须去除该前缀，只写 [Avatar:妖精爱莉]。内容连贯，不要直接出现"起因" "经过" "结果"这些词。
3.Users：选取1-5位今日最活跃或最有特色的群友（排除AI和系统消息），生成 RPG 风格的角色卡。stats 字段请填入 {"发言数": 0} 即可，真实数据将由代码自动填充。
4.Quotes：提取 3-5 条今日群内的搞笑、深刻或迷惑的发言（金句），忽略上下文缺失严重的短句。
5.内容风格：幽默、轻松、稍微带点二次元或游戏梗。

群聊记录：
"""
        
        messages: List[ChatCompletionMessageParam] = [{
            "role": "user",
            "content": prompt + f"\n{context}"
        }]
        
        # 尝试使用 JSON 模式 (如果模型支持)
        try:
            kwargs = {
                "model": self.model,
                "messages": messages,
                "temperature": 0.8,
                "response_format": {"type": "json_object"},
            }

            response = await self.client.chat.completions.create(**kwargs)
            
            return response.choices[0].message.content or "{}"
        except Exception:
            # 回退到普通模式
            return await self.generate_response(messages, retry)

class EmbeddingClient:
    """Embedding 模型客户端 - 专注文本向量化"""
    
    def __init__(self):
        config = ConfigManager()
        self.api_key = config.get_env("embedding_api_key")
        self.base_url = config.get_env("embedding_api_url")
        self.model = config.get_env("embedding_model_name", "Qwen/Qwen3-Embedding-8B")
        
        if not self.api_key or not self.base_url:
            logger.warning("Embedding Client 配置不完整，知识库功能可能受限。请检查 .env.dev 文件中的 embedding_api_key 和 embedding_api_url")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key or "dummy",
            base_url=self.base_url or "https://api.openai.com/v1"
        )
    
    async def get_embedding(self, text: str, retry: int = 3) -> List[float]:
        """获取文本的向量表示"""
        # 简单的输入检查
        if not text or not text.strip():
            return []

        for attempt in range(retry):
            try:
                # logger.debug(f"[Embedding] 获取向量: {text[:20]}...")
                response = await self.client.embeddings.create(
                    input=text,
                    model=self.model
                )
                return response.data[0].embedding
            except Exception as e:
                logger.error(f"[Embedding] 获取失败 (尝试 {attempt + 1}/{retry}): {type(e).__name__}: {str(e)}")
                await asyncio.sleep(1)
        
        return []
