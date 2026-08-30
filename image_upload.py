"""图片 Files API 后台上传与持久化重试。"""

import asyncio
import random
from contextlib import suppress
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from nonebot import logger

from .database import (
    ImageBatchCache,
    ImageFileCache,
    PendingImageBatch,
    PendingImageUpload,
)
from .logic import PreparedImageBatch, scoped_image_cache_key


PENDING_IMAGE_DIR = Path("data/Virtual_friends/pending_images")
INITIAL_UPLOAD_DELAY_SECONDS = 30
UPLOAD_RETRY_DELAYS = (5, 15, 45, 120, 300, 600, 1800, 3600)
UPLOAD_BATCH_SIZE = 10


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _write_image_file(path: Path, data: bytes) -> None:
    """原子写入待上传图片，避免重启后留下半个文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size == len(data):
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


class ImageUploadManager:
    """低优先级、单并发、可跨重启恢复的 Files API 上传器。"""

    def __init__(self, chat_client):
        self.chat_client = chat_client
        self.api_scope = getattr(chat_client, "file_cache_scope", "")
        self._task: Optional[asyncio.Task] = None
        self._wake_event = asyncio.Event()
        self._enqueue_lock = asyncio.Lock()
        self._process_lock = asyncio.Lock()
        self._stopping = False

    async def start(self) -> None:
        """启动后台循环，自动恢复数据库中的未完成任务。"""
        if self._task and not self._task.done():
            return
        PENDING_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
        # file_id 与上传时的 API Key 绑定；换配置后旧任务不能继续复用。
        if self.api_scope:
            await PendingImageBatch.exclude(api_scope=self.api_scope).delete()
            await PendingImageUpload.exclude(api_scope=self.api_scope).delete()
            await self._cleanup_spool()
        self._stopping = False
        self._task = asyncio.create_task(
            self._run(), name="virtual_friends_image_upload"
        )
        self._wake_event.set()
        pending_count = await PendingImageUpload.filter(
            api_scope=self.api_scope
        ).count()
        logger.info(f"[Vision] 后台上传器已启动，待处理 {pending_count} 张图片")

    async def stop(self) -> None:
        """停止后台循环；未完成任务保留在数据库中。"""
        self._stopping = True
        self._wake_event.set()
        if not self._task:
            return
        try:
            await asyncio.wait_for(self._task, timeout=15)
        except asyncio.TimeoutError:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
        finally:
            self._task = None

    async def enqueue(self, batch: PreparedImageBatch) -> None:
        """持久化图片组；当前回复无需等待 Files API。"""
        if not batch.images or not batch.cache_key:
            return

        async with self._enqueue_lock:
            now = _utc_now()
            batch_images: List[Dict[str, Any]] = []
            for image in batch.images:
                cache_key = scoped_image_cache_key(self.api_scope, image.md5)
                path = PENDING_IMAGE_DIR / image.filename
                batch_images.append(
                    {
                        "cache_key": cache_key,
                        "source_md5": image.md5,
                        "file_path": str(path),
                        "filename": image.filename,
                        "media_type": image.media_type,
                        "bytes": len(image.data),
                    }
                )

                if await ImageFileCache.get_or_none(md5=cache_key):
                    continue

                await asyncio.to_thread(_write_image_file, path, image.data)
                existing = await PendingImageUpload.get_or_none(md5=cache_key)
                if existing:
                    # 保留已经累积的重试次数，只修复可能缺失的本地文件信息。
                    existing.api_scope = self.api_scope
                    existing.source_md5 = image.md5
                    existing.file_path = str(path)
                    existing.filename = image.filename
                    existing.media_type = image.media_type
                    await existing.save(
                        update_fields=[
                            "api_scope",
                            "source_md5",
                            "file_path",
                            "filename",
                            "media_type",
                            "updated_at",
                        ]
                    )
                else:
                    await PendingImageUpload.create(
                        md5=cache_key,
                        api_scope=self.api_scope,
                        source_md5=image.md5,
                        file_path=str(path),
                        filename=image.filename,
                        media_type=image.media_type,
                        attempts=0,
                        next_retry_at=now
                        + timedelta(seconds=INITIAL_UPLOAD_DELAY_SECONDS),
                    )

            await PendingImageBatch.update_or_create(
                md5=batch.cache_key,
                defaults={
                    "api_scope": self.api_scope,
                    "images": batch_images,
                },
            )
            await self._finalize_ready_batches()
            self._wake_event.set()

    async def process_pending_once(self) -> int:
        """处理一小批到期任务，返回实际尝试数量。"""
        async with self._process_lock:
            now = _utc_now()
            rows = (
                await PendingImageUpload.filter(
                    api_scope=self.api_scope,
                    next_retry_at__lte=now,
                )
                .order_by("next_retry_at", "created_at")
                .limit(UPLOAD_BATCH_SIZE)
                .all()
            )
            for row in rows:
                await self._upload_one(row)
            await self._finalize_ready_batches()
            await self._cleanup_spool()
            return len(rows)

    async def _upload_one(self, row: PendingImageUpload) -> None:
        path = Path(row.file_path)
        try:
            image_data = await asyncio.to_thread(path.read_bytes)
            file_id = await self.chat_client.upload_image_file(
                image_data,
                row.filename,
                row.media_type,
                retry=1,
            )
            if not file_id:
                raise RuntimeError("Files API 上传未返回 file_id")
            await ImageFileCache.update_or_create(
                md5=row.md5,
                defaults={"file_id": file_id, "filename": row.filename},
            )
            await row.delete()
            logger.success(f"[Vision] 后台上传成功: filename={row.filename}")
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            attempts = row.attempts + 1
            delay_index = min(attempts - 1, len(UPLOAD_RETRY_DELAYS) - 1)
            base_delay = UPLOAD_RETRY_DELAYS[delay_index]
            delay = base_delay + random.uniform(0, max(1, base_delay * 0.2))
            row.attempts = attempts
            row.next_retry_at = _utc_now() + timedelta(seconds=delay)
            row.last_error = f"{type(exc).__name__}: {exc}"[:2000]
            await row.save(
                update_fields=[
                    "attempts",
                    "next_retry_at",
                    "last_error",
                    "updated_at",
                ]
            )
            logger.warning(
                f"[Vision] 后台上传失败，约 {int(delay)} 秒后重试: "
                f"filename={row.filename}, attempts={attempts}"
            )

    async def _finalize_ready_batches(self) -> None:
        rows = await PendingImageBatch.filter(api_scope=self.api_scope).all()
        for row in rows:
            if not isinstance(row.images, list) or not row.images:
                await row.delete()
                continue

            cache_keys = [
                item.get("cache_key")
                for item in row.images
                if isinstance(item, dict) and isinstance(item.get("cache_key"), str)
            ]
            if len(cache_keys) != len(row.images):
                logger.warning(f"[Vision] 待上传图片组数据损坏: {row.md5}")
                continue
            cached_rows = await ImageFileCache.filter(md5__in=cache_keys).all()
            cached_map = {cached.md5: cached for cached in cached_rows}
            if any(cache_key not in cached_map for cache_key in cache_keys):
                continue

            files: List[Dict[str, Any]] = []
            for item in row.images:
                cache_key = item["cache_key"]
                files.append(
                    {
                        "file_id": cached_map[cache_key].file_id,
                        "bytes": item.get("bytes", 0),
                    }
                )
            await ImageBatchCache.update_or_create(
                md5=row.md5,
                defaults={"api_scope": self.api_scope, "files": files},
            )
            await row.delete()
            logger.debug(f"[Vision] 图片组永久缓存已就绪: {row.md5}")

    async def _cleanup_spool(self) -> None:
        """只删除已经不被上传任务或待完成图片组引用的本地文件。"""
        referenced: Set[str] = set(
            await PendingImageUpload.all().values_list("file_path", flat=True)
        )
        for row in await PendingImageBatch.all():
            if not isinstance(row.images, list):
                continue
            referenced.update(
                item.get("file_path")
                for item in row.images
                if isinstance(item, dict) and isinstance(item.get("file_path"), str)
            )

        if not PENDING_IMAGE_DIR.exists():
            return
        for path in PENDING_IMAGE_DIR.iterdir():
            if not path.is_file() or str(path) in referenced:
                continue
            try:
                await asyncio.to_thread(path.unlink)
            except OSError as exc:
                logger.warning(f"[Vision] 清理本地图片缓存失败 {path}: {exc}")

    async def _next_wait_timeout(self, processed: int) -> float:
        if processed >= UPLOAD_BATCH_SIZE:
            return 0.1
        next_row = (
            await PendingImageUpload.filter(api_scope=self.api_scope)
            .order_by("next_retry_at")
            .first()
        )
        if not next_row:
            return 30.0
        next_retry_at = next_row.next_retry_at
        if next_retry_at.tzinfo is None:
            next_retry_at = next_retry_at.replace(tzinfo=timezone.utc)
        remaining = (next_retry_at - _utc_now()).total_seconds()
        return min(30.0, max(0.1, remaining))

    async def _run(self) -> None:
        while not self._stopping:
            try:
                processed = await self.process_pending_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                processed = 0
                logger.exception("[Vision] 后台上传循环异常")

            self._wake_event.clear()
            timeout = await self._next_wait_timeout(processed)
            try:
                await asyncio.wait_for(self._wake_event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                pass
