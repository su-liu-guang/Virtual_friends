import json
import re
import math
import asyncio
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Set
from nonebot import logger
from .clients import EmbeddingClient
from .config import ConfigManager
from .memory_retriever import RerankerClient

class KnowledgeBase:
    def __init__(self):
        self.base_path = Path("data/Virtual_friends/knowledge")
        self.index_path = self.base_path / "knowledge_index.json"
        self.embedding_client = EmbeddingClient()
        self.reranker = RerankerClient(ConfigManager())
        
        # 内存中的索引
        self.chunks: List[Dict] = []
        self.image_map: Dict[str, str] = {} 
        self.file_hashes: Dict[str, str] = {} # 记录文件哈希 relative_path -> md5
        
        # 确保目录存在
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """初始化知识库：加载索引并执行增量更新"""
        # 1. 加载现有索引
        if self.index_path.exists():
            try:
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                self.chunks = data.get("chunks", [])
                self.image_map = data.get("image_map", {})
                self.file_hashes = data.get("file_hashes", {})
                logger.info(f"加载现有索引: {len(self.chunks)} 个片段, {len(self.image_map)} 张图片")
            except Exception as e:
                logger.error(f"加载索引失败，将重建: {e}")
                self.chunks = []
                self.image_map = {}
                self.file_hashes = {}

        # 2. 扫描磁盘文件并执行增量更新
        await self._incremental_update()

    async def _incremental_update(self):
        """增量更新索引"""
        # 获取所有 .md 文件
        all_md_files = list(self.base_path.rglob("*.md"))
        
        # 过滤掉隐藏目录（如 .git, .github）和特定文件
        disk_files = []
        ignored_files = {"LICENSE.md", "CONTRIBUTING.md", "CHANGELOG.md", "CODE_OF_CONDUCT.md"}
        
        for f in all_md_files:
            try:
                rel_path = f.relative_to(self.base_path)
                # 1. 排除隐藏目录 (.git, .github 等)
                if any(part.startswith('.') for part in rel_path.parts):
                    continue
                # 2. 排除特定文件名
                if f.name in ignored_files:
                    continue
                disk_files.append(f)
            except ValueError:
                continue

        if not disk_files:
            if self.chunks:
                logger.warning("磁盘上未找到有效文档，清空索引")
                self.chunks = []
                self.image_map = {}
                self.file_hashes = {}
                self._save_index()
            return

        current_files_status = {} # relative_path -> hash
        files_to_process = []
        
        # 1. 计算当前所有文件的哈希
        for file_path in disk_files:
            try:
                rel_path = str(file_path.relative_to(self.base_path))
                file_hash = self._calculate_file_hash(file_path)
                current_files_status[rel_path] = file_hash
                
                # 检查是否需要更新
                if rel_path not in self.file_hashes or self.file_hashes[rel_path] != file_hash:
                    files_to_process.append(file_path)
            except Exception as e:
                logger.error(f"处理文件 {file_path} 出错: {e}")

        # 2. 找出需要删除的文件 (在索引中但不在磁盘上)
        files_to_remove = set(self.file_hashes.keys()) - set(current_files_status.keys())
        
        if not files_to_process and not files_to_remove:
            logger.success("知识库已是最新，无需更新")
            return

        logger.info(f"检测到变更: {len(files_to_process)} 个文件更新/新增, {len(files_to_remove)} 个文件删除")

        # 3. 清理旧数据
        if files_to_remove or files_to_process:
            # 需要移除的源文件路径集合 (包括被删除的文件 和 需要重新处理的文件)
            sources_to_clear = files_to_remove.union({str(f.relative_to(self.base_path)) for f in files_to_process})
            
            # 过滤 chunks
            original_count = len(self.chunks)
            self.chunks = [c for c in self.chunks if c.get("source") not in sources_to_clear]
            logger.info(f"清理旧索引: {original_count} -> {len(self.chunks)} (移除 {original_count - len(self.chunks)} 个片段)")
            
            # 更新哈希记录
            for rel_path in files_to_remove:
                self.file_hashes.pop(rel_path, None)

        # 4. 处理新文件
        for file_path in files_to_process:
            await self._process_file(file_path)
            # 更新哈希记录
            rel_path = str(file_path.relative_to(self.base_path))
            self.file_hashes[rel_path] = current_files_status[rel_path]

        # 5. 保存索引
        self._save_index()
        logger.success(f"增量更新完成: 当前共 {len(self.chunks)} 个片段")

    def _calculate_file_hash(self, file_path: Path) -> str:
        """计算文件 MD5"""
        return hashlib.md5(file_path.read_bytes()).hexdigest()

    async def rebuild_index(self):
        """强制重建索引 (已废弃，保留兼容性，实际调用增量更新)"""
        self.chunks = []
        self.image_map = {}
        self.file_hashes = {}
        await self._incremental_update()

    async def _process_file(self, file_path: Path):
        logger.info(f"正在处理文档: {file_path.name}")
        try:
            content = file_path.read_text(encoding="utf-8")
            rel_source_path = str(file_path.relative_to(self.base_path))
            
            # 1. 提取并替换图片
            def replace_image(match):
                alt = match.group(1)
                rel_path = match.group(2)
                
                # 计算绝对路径
                try:
                    img_abs_path = (file_path.parent / rel_path).resolve()
                    if img_abs_path.exists():
                        img_key = alt if alt else img_abs_path.stem
                        self.image_map[img_key] = str(img_abs_path)
                        return f"(此处有一张图片，名称为：{img_key})"
                except Exception:
                    pass
                return f"(图片丢失: {rel_path})"

            content = re.sub(r"!\[(.*?)\]\((.*?)\)", replace_image, content)
            
            # 2. 切分文档 (按标题)
            lines = content.split('\n')
            current_chunk = []
            current_title = "导言"
            
            chunks_text = []
            
            for line in lines:
                if line.strip().startswith('#'):
                    if current_chunk:
                        chunks_text.append((current_title, "\n".join(current_chunk)))
                    current_title = line.strip().lstrip('#').strip()
                    current_chunk = [line]
                else:
                    current_chunk.append(line)
                    
            if current_chunk:
                chunks_text.append((current_title, "\n".join(current_chunk)))
                
            # 3. 向量化并存储
            for title, text in chunks_text:
                if not text.strip():
                    continue
                    
                embedding = await self.embedding_client.get_embedding(text)
                if embedding:
                    self.chunks.append({
                        "title": title,
                        "text": text,
                        "source": rel_source_path, # 使用相对路径作为 source
                        "embedding": embedding
                    })
        except Exception as e:
            logger.error(f"处理文件 {file_path} 失败: {e}")

    def _save_index(self):
        data = {
            "chunks": self.chunks,
            "image_map": self.image_map,
            "file_hashes": self.file_hashes
        }
        self.index_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


    async def search(self, query: str, top_k: int = 3, min_score: float = 0.28) -> List[Dict]:
        """搜索相关片段，低于阈值则视为不相关，避免乱插参考资料；可选 rerank 优化排序"""
        if not self.chunks:
            return []
            
        query_embedding = await self.embedding_client.get_embedding(query)
        if not query_embedding:
            return []
            
        # 计算相似度
        scored_chunks = []
        for chunk in self.chunks:
            score = self._cosine_similarity(query_embedding, chunk["embedding"])
            scored_chunks.append((score, chunk))
            
        # 基于向量的初步排序
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        if not scored_chunks:
            return []

        # 阈值过滤
        best_score = scored_chunks[0][0]
        if best_score < min_score:
            logger.debug(
                f"[Knowledge] 最高相似度 {best_score:.3f} < 阈值 {min_score}, 不插入参考资料"
            )
            return []

        # 可选重排序
        if self.reranker.enabled:
            candidates = scored_chunks[: max(top_k * 3, 6)]
            docs = [c[1]["text"] for c in candidates]
            logger.debug(
                f"[Knowledge] Reranker 启用，候选={len(docs)} query_len={len(query)}"
            )
            rerank_scores = await self.reranker.rerank(query, docs)
            if rerank_scores and len(rerank_scores) == len(docs):
                logger.info("[Knowledge] Reranker 成功，采用重排序结果")
                reranked = sorted(
                    zip(rerank_scores, candidates), key=lambda x: x[0], reverse=True
                )
                scored_chunks = [item[1] for item in reranked]
            else:
                logger.warning("[Knowledge] Reranker 失败或返回数量不匹配，沿用向量排序")
        else:
            logger.debug("[Knowledge] Reranker 未启用，沿用向量排序")

        return [item[1] for item in scored_chunks[:top_k]]

    def _cosine_similarity(self, v1: List[float], v2: List[float]) -> float:
        """计算余弦相似度"""
        dot_product = sum(a * b for a, b in zip(v1, v2))
        norm_a = math.sqrt(sum(a * a for a in v1))
        norm_b = math.sqrt(sum(b * b for b in v2))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot_product / (norm_a * norm_b)

    def get_image_path(self, name: str) -> Optional[str]:
        return self.image_map.get(name)
