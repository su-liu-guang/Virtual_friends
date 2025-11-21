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
        
        # 打印配置信息
        logger.info("=" * 50)
        logger.info("Vision Client 配置:")
        logger.info(f"  API URL: {self.base_url or '未配置'}")
        logger.info(f"  API Key: {self.api_key[:10]}...{self.api_key[-4:] if len(self.api_key) > 14 else ''}" if self.api_key else "  API Key: 未配置")
        logger.info(f"  Model: {self.model}")
        logger.info("=" * 50)
        
        if not self.api_key or not self.base_url:
            logger.error("Vision Client 配置不完整，请检查 .env.dev 文件中的 vision_api_key 和 vision_api_url")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key or "dummy",
            base_url=self.base_url or "https://api.openai.com/v1"
        )
    
    async def recognize_image(self, image_url: str, retry: int = 3, is_sticker: bool = False) -> str:
        """识别图片内容"""
        logger.info(f"[Vision] 开始识别图片: {image_url[:50]}... (表情包: {is_sticker})")
        
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
                
                logger.success(f"[Vision] 识别成功: {result[:100]}...")
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
        
        # 打印配置信息
        logger.info("=" * 50)
        logger.info("Chat Client 配置:")
        logger.info(f"  API URL: {self.base_url or '未配置'}")
        logger.info(f"  API Key: {self.api_key[:10]}...{self.api_key[-4:] if len(self.api_key) > 14 else ''}" if self.api_key else "  API Key: 未配置")
        logger.info(f"  Model: {self.model}")
        logger.info("=" * 50)
        
        if not self.api_key or not self.base_url:
            logger.error("Chat Client 配置不完整，请检查 .env.dev 文件中的 chat_api_key 和 chat_api_url")
        
        self.client = AsyncOpenAI(
            api_key=self.api_key or "dummy",
            base_url=self.base_url or "https://api.openai.com/v1"
        )
    
    async def generate_response(self, messages: Sequence[ChatCompletionMessageParam], retry: int = 3) -> str:
        """生成对话回复"""
        logger.info(f"[Chat] 开始生成回复, 上下文消息数: {len(messages)}")
        logger.debug(f"[Chat] 消息预览:")
        for i, msg in enumerate(messages[:3]):  # 只显示前3条
            role = msg.get("role", "unknown")
            content = str(msg.get("content", ""))[:100]
            logger.debug(f"  [{i}] {role}: {content}...")
        
        for attempt in range(retry):
            try:
                logger.debug(f"[Chat] 尝试 {attempt + 1}/{retry}")
                logger.debug(f"[Chat] 调用参数: model={self.model}, temperature=0.8")
                
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.8
                )
                
                result = response.choices[0].message.content or "..."
                logger.success(f"[Chat] 生成成功: {result[:100]}...")
                logger.debug(f"[Chat] Token 使用: prompt={response.usage.prompt_tokens if response.usage else 'N/A'}, completion={response.usage.completion_tokens if response.usage else 'N/A'}, total={response.usage.total_tokens if response.usage else 'N/A'}")
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
            "content": f"总结以下对话,忽略寒暄和无关细节,保留主要事件和讨论点:\n\n{context}"
        }]
        return await self.generate_response(messages, retry)
    
    async def extract_facts(self, context: str, retry: int = 3) -> List[str]:
        """提取重要事实"""
        logger.info(f"[Chat] 开始提取事实, 上下文长度: {len(context)}")
        messages: List[ChatCompletionMessageParam] = [{
            "role": "user",
            "content": f"从以下对话中提取需要长期记忆的用户信息、计划或重要事实。一定要是非常重要的内容,每行一条,无则返回'无',宁缺毋滥:\n\n{context}"
        }]
        
        result = await self.generate_response(messages, retry)
        if result.strip() == "无":
            logger.info("[Chat] 未提取到任何事实")
            return []
        
        facts = [line.strip() for line in result.split("\n") if line.strip() and not line.startswith("#")]
        logger.success(f"[Chat] 提取到 {len(facts)} 条事实")
        return facts
