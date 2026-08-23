import asyncio
import hashlib
import io
import aiohttp
import re
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple
from nonebot import logger
from openai.types.chat import ChatCompletionMessageParam
from PIL import Image, UnidentifiedImageError
from .database import ImageBatchCache, ImageFileCache, Message, Summary
from .config import ConfigManager

MAX_IMAGES_PER_MESSAGE = 9
MAX_IMAGES_PER_CONTEXT = 14
MAX_IMAGE_BYTES = 64 * 1024 * 1024
MAX_TOTAL_IMAGE_BYTES = 200 * 1024 * 1024
MAX_IMAGE_DIMENSION = 8192
IMAGE_DOWNLOAD_TIMEOUT_SECONDS = 15
SUPPORTED_IMAGE_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "GIF": "image/gif",
    "WEBP": "image/webp",
}


@dataclass(frozen=True)
class ImageSource:
    """来自 OneBot 消息段的图片输入。"""

    url: str
    is_sticker: bool = False


@dataclass(frozen=True)
class PreparedImage:
    """已下载并校验、等待或已经上传到 Files API 的图片。"""

    md5: str
    data: bytes
    filename: str
    media_type: str
    width: int
    height: int
    is_sticker: bool
    file_id: Optional[str] = None

    def to_content_block(self) -> Dict[str, Any]:
        if not self.file_id:
            raise ValueError("图片尚未上传到 Files API")
        return {"type": "file", "file_id": self.file_id}


@dataclass(frozen=True)
class PreparedImageBatch:
    """一条消息中通过校验的图片集合。"""

    images: Tuple[PreparedImage, ...]
    cache_key: Optional[str]
    failed_count: int = 0

    @property
    def content_blocks(self) -> List[Dict[str, Any]]:
        return [image.to_content_block() for image in self.images]

# ====== 知识库命中时注入的精确输出指令（优先级高于人设）======
KNOWLEDGE_PRECISION_INSTRUCTION = """[资料库精确输出指令 - 优先级高于人设]

检测到知识库中存在与该问题相关的参考资料，请严格遵守以下规则：
1. 将参考资料中所有相关信息**完整、逐条**输出，不得省略关键细节
2. 严格基于参考资料原文，**禁止编造或补充**参考资料中不存在的信息
3. 如果资料包含具体数据、名称、步骤，必须全部包含在回复中
4. 上述精确输出要求**高于人设聊天风格**——资料完整性优先于角色扮演
5. 如果资料不足，明确说明，不要编造""".strip()

# ====== 录取资料格式约束（临时硬编码，past_conditions.md 命中时注入）======
PAST_CONDITIONS_FORMAT_RULES = """[录取数据输出格式 - 强制遵守]

回答河南科技大学录取数据时，必须严格遵守以下规则：
1. 必须注明**年份**（2024年或2025年）和**最低分位次**（或最低分排位）
2. 如果是**专项计划**（国家专项、地方专项），必须明确标注「该数据为[国家/地方]专项计划录取数据」
3. **禁止提及专业组编号**（如"102组""301组"），专业组编号对用户无意义
4. 先列出普通批次数据，再列出专项计划数据

[输出示例]
根据2024年和2025年录取数据，xxx专业的最低分位次如下：
**2025年：**
- 普通批：最低分位次 90232
- 国家专项计划：最低分位次 47010
- 地方专项计划：最低分位次 47010
- 本科提前批其他类：最低分位次 119532
**2024年：**
- 本科第一批：最低分排位 90628
- 国家专项计划：最低分位次 47010
- 地方专项计划：最低分位次 47010
- 本科提前批：最低分排位 106610

如果没有专项计划或提前批则不输出对应内容,所有数据必须真实,禁止输出没有依据的数据
""".strip()

def sanitize_reply(raw_text: Optional[str]) -> str:
    """清洗模型输出中的常见元信息。"""
    if not raw_text:
        return ""

    text = raw_text.strip()
    if not text:
        return ""

    # 去掉常见 markdown 包裹
    text = re.sub(r"^```(?:[a-zA-Z]+)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    # 去除可能的时间戳前缀 [2025-12-12 19:30]
    text = re.sub(r"^\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}\]\s*", "", text)

    # 逐行移除常见元信息和说话人前缀
    cleaned_lines: List[str] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue

        if re.match(r"^\{\{发送图片:.*\}\}$", line):
            cleaned_lines.append(line)
            continue

        if re.match(r"^(?:说明|注[:：]|系统|旁白|内心|思考|analysis|assistant|user)\b", line, flags=re.IGNORECASE):
            continue

        line = re.sub(
            r"^(?:\[[^\]]{1,20}\]|（[^）]{1,20}）|\([^)]{1,20}\)|[A-Za-z\u4e00-\u9fa5]{1,20})\s*[：:]\s*",
            "",
            line,
        )
        if line:
            cleaned_lines.append(line)

    result = "\n".join(cleaned_lines).strip()
    result = re.sub(r"（[^）]*）", "", result)
    result = re.sub(r"\([^)]*\)", "", result)
    result = re.sub(r"\s{2,}", " ", result).strip()
    return result


def _inspect_image(image_data: bytes) -> Tuple[str, str, int, int]:
    """按文件实际内容识别格式与尺寸。"""
    try:
        with Image.open(io.BytesIO(image_data)) as image:
            image_format = (image.format or "").upper()
            width, height = image.size
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("无法识别图片内容") from exc

    media_type = SUPPORTED_IMAGE_FORMATS.get(image_format)
    if not media_type:
        raise ValueError(f"不支持的图片格式: {image_format or '未知'}")
    if width <= 0 or height <= 0:
        raise ValueError("图片尺寸无效")
    if max(width, height) > MAX_IMAGE_DIMENSION:
        raise ValueError(
            f"图片单边超过 {MAX_IMAGE_DIMENSION} 像素: {width}x{height}"
        )
    return image_format, media_type, width, height


async def _download_image(
    session: aiohttp.ClientSession,
    source: ImageSource,
    byte_limit: int,
) -> bytes:
    if not source.url.startswith(("http://", "https://")):
        raise ValueError("图片 URL 必须使用 http(s)")
    if byte_limit <= 0:
        raise ValueError("本条消息的图片总大小已达上限")

    async with session.get(source.url) as response:
        response.raise_for_status()
        content_length = response.content_length
        if content_length is not None and content_length > byte_limit:
            raise ValueError(f"图片超过剩余大小限制: {content_length} bytes")

        chunks: List[bytes] = []
        total = 0
        async for chunk in response.content.iter_chunked(64 * 1024):
            total += len(chunk)
            if total > byte_limit:
                raise ValueError(f"图片超过大小限制: > {byte_limit} bytes")
            chunks.append(chunk)
        if not chunks:
            raise ValueError("图片内容为空")
        return b"".join(chunks)


def _build_image_batch(
    images: Sequence[PreparedImage], failed_count: int = 0
) -> PreparedImageBatch:
    if not images:
        return PreparedImageBatch((), None, failed_count)
    if len(images) == 1:
        cache_key = images[0].md5
    else:
        ordered_digests = "\0".join(image.md5 for image in images)
        cache_key = hashlib.md5(ordered_digests.encode("ascii")).hexdigest()
    return PreparedImageBatch(tuple(images), cache_key, failed_count)


async def prepare_image_messages(
    sources: Sequence[ImageSource],
    user_text: str = "",
) -> PreparedImageBatch:
    """下载并准备一条消息中的最多 9 张图片。"""
    selected_sources = list(sources[:MAX_IMAGES_PER_MESSAGE])
    ignored_count = max(0, len(sources) - len(selected_sources))
    if ignored_count:
        logger.warning(
            f"[Vision] 单条消息图片超过 {MAX_IMAGES_PER_MESSAGE} 张，"
            f"已忽略后 {ignored_count} 张"
        )

    prepared: List[PreparedImage] = []
    failed_count = 0
    total_bytes = 0
    timeout = aiohttp.ClientTimeout(total=IMAGE_DOWNLOAD_TIMEOUT_SECONDS)

    async with aiohttp.ClientSession(timeout=timeout) as session:
        for index, source in enumerate(selected_sources, start=1):
            try:
                remaining = MAX_TOTAL_IMAGE_BYTES - total_bytes
                image_data = await _download_image(
                    session, source, min(MAX_IMAGE_BYTES, remaining)
                )
                # Pillow 只解析文件头并校验结构，不执行像素级处理。
                image_format, media_type, width, height = _inspect_image(image_data)
                digest = hashlib.md5(image_data).hexdigest()
                extension = "jpg" if image_format == "JPEG" else image_format.lower()
                prepared.append(
                    PreparedImage(
                        md5=digest,
                        data=image_data,
                        filename=f"{digest}.{extension}",
                        media_type=media_type,
                        width=width,
                        height=height,
                        is_sticker=source.is_sticker,
                    )
                )
                total_bytes += len(image_data)
                logger.debug(
                    f"[Vision] 图{index}已准备: format={image_format}, "
                    f"size={width}x{height}, bytes={len(image_data)}"
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                failed_count += 1
                logger.warning(f"[Vision] 图{index}准备失败，已跳过: {exc}")

    return _build_image_batch(prepared, failed_count)


async def upload_image_files(batch: PreparedImageBatch, chat_client) -> PreparedImageBatch:
    """复用永久 file_id，未命中时上传；单图失败不影响其余图片。"""
    if not batch.images:
        return batch

    uploaded_images: List[PreparedImage] = []
    failed_count = batch.failed_count
    api_scope = getattr(chat_client, "file_cache_scope", "")

    for index, image in enumerate(batch.images, start=1):
        try:
            # DeepSeek file_id 归属于上传时使用的 API Key，换 Key 后必须重传。
            scoped_cache_key = hashlib.md5(
                f"{api_scope}\0{image.md5}".encode("ascii")
            ).hexdigest()
            cached = await ImageFileCache.get_or_none(md5=scoped_cache_key)
            if cached:
                uploaded_images.append(replace(image, file_id=cached.file_id))
                logger.debug(f"[Vision] 图{index} Files API 缓存命中")
                continue

            result = await chat_client.upload_image_file(
                image.data,
                image.filename,
                image.media_type,
            )
            if not result:
                raise RuntimeError("Files API 上传未返回结果")
            file_id = result
            await ImageFileCache.update_or_create(
                md5=scoped_cache_key,
                defaults={
                    "file_id": file_id,
                    "filename": image.filename,
                },
            )
            uploaded_images.append(replace(image, file_id=file_id))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            failed_count += 1
            logger.warning(f"[Vision] 图{index}上传失败，已跳过: {exc}")

    uploaded_batch = _build_image_batch(uploaded_images, failed_count)
    if uploaded_batch.images and uploaded_batch.cache_key:
        await ImageBatchCache.update_or_create(
            md5=uploaded_batch.cache_key,
            defaults={
                "api_scope": api_scope,
                "files": [
                    {
                        "file_id": image.file_id,
                        "bytes": len(image.data),
                    }
                    for image in uploaded_batch.images
                ],
            },
        )
    return uploaded_batch


async def collect_message_image_blocks(
    messages: Sequence[Any],
    api_scope: str,
) -> Dict[Any, List[Dict[str, Any]]]:
    """为一组消息选取可直接发送的永久图片，优先保留较新的图片。"""
    cache_keys = {
        msg.image_md5
        for msg in messages
        if getattr(msg, "role", None) == "user" and getattr(msg, "image_md5", None)
    }
    if not cache_keys:
        return {}

    rows = await ImageBatchCache.filter(md5__in=cache_keys).all()
    cache_map = {
        row.md5: row
        for row in rows
        if row.api_scope == api_scope and isinstance(row.files, list)
    }

    selected: Dict[Any, List[Dict[str, Any]]] = {}
    image_count = 0
    total_bytes = 0
    for msg in reversed(messages):
        if getattr(msg, "role", None) != "user":
            continue
        row = cache_map.get(getattr(msg, "image_md5", None))
        if not row:
            continue

        blocks: List[Dict[str, Any]] = []
        for item in row.files:
            if not isinstance(item, dict):
                continue
            file_id = item.get("file_id")
            file_bytes = item.get("bytes", 0)
            if not isinstance(file_id, str) or not file_id:
                continue
            if not isinstance(file_bytes, int) or file_bytes < 0:
                continue
            if image_count >= MAX_IMAGES_PER_CONTEXT:
                break
            if total_bytes + file_bytes > MAX_TOTAL_IMAGE_BYTES:
                continue
            blocks.append({"type": "file", "file_id": file_id})
            image_count += 1
            total_bytes += file_bytes
        if blocks:
            selected[msg.id] = blocks
        if image_count >= MAX_IMAGES_PER_CONTEXT:
            break
    return selected


class ContextBuilder:
    """上下文构建器"""
    
    def __init__(
        self,
        config_manager: ConfigManager,
        knowledge_base=None,
        file_cache_scope: str = "",
    ):
        self.config_manager = config_manager
        self.knowledge_base = knowledge_base
        self.file_cache_scope = file_cache_scope
    
    async def build_context(
        self, 
        group_id: str, 
        user_nickname: str, 
        current_time: datetime,
        current_images: Optional[Sequence[Dict[str, Any]]] = None,
    ) -> List[ChatCompletionMessageParam]:
        """构建完整上下文，永久图片只附加到对应的 user 消息。"""
        config = self.config_manager.get_instance_config(group_id)
        persona_prompt = self.config_manager.get_persona_prompt(config["persona_name"])
        
        # 当前未处理消息在摘要周期内只会追加，不使用会逐条滑动的窗口。
        recent_msgs = await Message.filter(
            group_id=group_id,
            is_processed=False,
        ).order_by("timestamp", "id").all()
        historical_image_blocks = (
            await collect_message_image_blocks(recent_msgs, self.file_cache_scope)
            if self.file_cache_scope
            else {}
        )

        # 归档阈值保证未归档 L1<20、L2<10，因此可完整加载。
        all_l1 = (
            await Summary.filter(group_id=group_id, level=1, is_archived=False)
            .order_by("created_at", "id")
            .all()
        )
        all_l2 = (
            await Summary.filter(group_id=group_id, level=2, is_archived=False)
            .order_by("created_at", "id")
            .all()
        )
        latest_l3 = (
            await Summary.filter(group_id=group_id, level=3, is_archived=False)
            .order_by("-created_at", "-id")
            .limit(5)
            .all()
        )
        all_l3 = list(reversed(latest_l3))

        # ---- Knowledge Layer ----
        knowledge_text = ""
        is_past_conditions = False
        last_user_msg = None
        if recent_msgs:
            for msg in reversed(recent_msgs):
                if msg.role == "user":
                    last_user_msg = msg
                    break

        if self.knowledge_base and last_user_msg and last_user_msg.content and len(last_user_msg.content) > 2:
            knowledge_query = last_user_msg.content
            chunks = await self.knowledge_base.search(knowledge_query, top_k=5)
            if chunks:
                knowledge_text = "\n[参考资料]:\n"
                for i, chunk in enumerate(chunks):
                    knowledge_text += f"{i+1}. (来自 {chunk['source']} - {chunk['title']})\n{chunk['text']}\n"
                
                knowledge_text += "\n[图片发送规则]\n参考资料中可能会提到图片资源，格式为 `(此处有一张图片，名称为：xxx)`。\n如果你认为展示该图片有助于回答用户问题，请在回复的末尾单独一行输出：`{{发送图片:xxx}}`。\n请只发送与问题高度相关的图片。\n"

                # ---- 临时：past_conditions.md 命中时注入格式约束 ----
                is_past_conditions = any(
                    c.get("source") == "past_conditions.md" for c in chunks
                )

        # ====== SYSTEM MESSAGE: 人设 + 分层记忆 ======
        section_persona = persona_prompt
        section_l3 = ""
        section_l2 = ""
        section_l1 = ""

        # L3 宏观印象 — 数月不变，准静态
        if all_l3:
            section_l3 = "\n\n[宏观印象 L3]:\n"
            for s3 in all_l3:
                section_l3 += f"- ({s3.time_range}) {s3.content.strip()}\n"

        # L2 叙事概括 — 数天不变，相对稳定
        if all_l2:
            section_l2 = "\n\n[叙事概括 L2]:\n"
            for s2 in all_l2:
                section_l2 += f"- ({s2.time_range}) {s2.content.strip()}\n"

        # L1 近期详情 — 每 50 条消息变化一次，49/50 次可命中缓存
        if all_l1:
            section_l1 = "\n\n[近期详情 L1]:\n"
            for s1 in all_l1:
                section_l1 += f"- ({s1.time_range}) {s1.content.strip()}\n"

        system_content = section_persona + section_l3 + section_l2 + section_l1

        messages: List[ChatCompletionMessageParam] = [{"role": "system", "content": system_content.strip()}]
        
        # ---- Recent Layer ----
        for msg in recent_msgs:
            content = msg.api_content if msg.role == "ai" and msg.api_content else msg.content

            is_current_image_message = bool(
                current_images and last_user_msg and msg.id == last_user_msg.id
            )
            image_blocks = (
                [dict(block) for block in current_images]
                if is_current_image_message
                else historical_image_blocks.get(msg.id, [])
            )
            
            local_ts = msg.timestamp.astimezone()
            time_prefix = f"[{local_ts.strftime('%Y-%m-%d %H:%M')}] "
            
            if msg.role == "user" and msg.user_nickname:
                content = f"{time_prefix}{msg.user_nickname}: {content}" if content else f"{time_prefix}{msg.user_nickname}: [无文本内容]"
            role = "assistant" if msg.role == "ai" else "user"
            message: ChatCompletionMessageParam = {
                "role": role,  # type: ignore
                "content": content
            }
            if role == "user" and image_blocks:
                message["content"] = [  # type: ignore[index]
                    {"type": "text", "text": content},
                    *image_blocks,
                ]
            if msg.role == "ai" and msg.reasoning_content:
                message["reasoning_content"] = msg.reasoning_content  # type: ignore[index]
            messages.append(message)

        # 知识库属于本次检索结果，作为独立临时消息追加，不改写持久化用户消息。
        section_knowledge = ""
        section_knowledge_instr = ""
        if knowledge_text.strip():
            section_knowledge = knowledge_text.strip()
            precision = KNOWLEDGE_PRECISION_INSTRUCTION
            if is_past_conditions:
                precision += "\n\n" + PAST_CONDITIONS_FORMAT_RULES
            section_knowledge_instr = precision
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "[本次请求的知识库参考资料]\n"
                        + section_knowledge
                        + "\n\n"
                        + section_knowledge_instr
                    ),
                }
            )

        # ====== DEBUG: 各部分长度测量 ======
        _total_system = len(section_persona) + len(section_l3) + len(section_l2) + len(section_l1)
        _recent_total = 0
        for message in messages:
            if message["role"] not in ("user", "assistant"):
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                _recent_total += len(content)
            elif isinstance(content, list):
                _recent_total += sum(
                    len(block.get("text", ""))
                    for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
        _total_all = _total_system + _recent_total
        logger.info(
            f"[Cache Profiler] group={group_id} total_chars={_total_all} | "
            f"persona={len(section_persona)} "
            f"L3={len(section_l3)} L2={len(section_l2)} L1={len(section_l1)} "
            f"sys_total={_total_system} ({_total_system*100//max(_total_all,1)}%) | "
            f"recent_msgs={_recent_total} ({_recent_total*100//max(_total_all,1)}%) "
            f"count={len(recent_msgs)} | "
            f"knowledge={len(section_knowledge)} "
            f"k_instr={len(section_knowledge_instr)}"
        )

        return messages
