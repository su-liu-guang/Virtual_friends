import json
import re
import math
import asyncio
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from nonebot import logger
from .clients import EmbeddingClient

class KnowledgeBase:
    def __init__(self):
        self.base_path = Path("data/Virtual_friends/knowledge")
        self.index_path = self.base_path / "knowledge_index.json"
        self.embedding_client = EmbeddingClient()
        
        # 内存中的索引
        # chunks: List[Dict] = [{"text": "...", "embedding": [...], "source": "file.md", "images": {"name": "path"}}]
        self.chunks: List[Dict] = []
        self.image_map: Dict[str, str] = {} # 全局图片映射 name -> absolute_path
        
        # 确保目录存在
        self.base_path.mkdir(parents=True, exist_ok=True)

    async def initialize(self):
        """初始化知识库：加载索引或重新构建"""
        if self.index_path.exists():
            try:
                logger.info("正在加载知识库索引...")
                data = json.loads(self.index_path.read_text(encoding="utf-8"))
                self.chunks = data.get("chunks", [])
                self.image_map = data.get("image_map", {})
                logger.success(f"知识库加载完成: {len(self.chunks)} 个片段, {len(self.image_map)} 张图片")
            except Exception as e:
                logger.error(f"加载索引失败，将重建: {e}")
                await self.rebuild_index()
        else:
            logger.info("知识库索引不存在，开始构建...")
            await self.rebuild_index()

    async def rebuild_index(self):
        """重建索引"""
        self.chunks = []
        self.image_map = {}
        
        # 递归查找所有 .md 文件
        md_files = list(self.base_path.rglob("*.md"))
        if not md_files:
            logger.warning("未找到任何 Markdown 文档")
            return

        logger.info(f"找到 {len(md_files)} 个文档，开始处理...")
        
        for file_path in md_files:
            await self._process_file(file_path)
            
        # 保存索引
        self._save_index()
        logger.success(f"索引构建完成: {len(self.chunks)} 个片段")

    async def _process_file(self, file_path: Path):
        content = file_path.read_text(encoding="utf-8")
        
        # 1. 提取并替换图片
        # Markdown 图片格式: ![alt](path)
        # 我们假设 path 是相对路径
        
        def replace_image(match):
            alt = match.group(1)
            rel_path = match.group(2)
            
            # 计算绝对路径
            # 图片通常在文档同级或子目录
            img_abs_path = (file_path.parent / rel_path).resolve()
            
            if img_abs_path.exists():
                # 记录映射
                # 为了避免重名，可以使用 "文件名_alt" 或者直接用 alt (如果用户保证唯一)
                # 这里简单起见，使用 alt，如果 alt 为空则使用文件名
                img_key = alt if alt else img_abs_path.stem
                self.image_map[img_key] = str(img_abs_path)
                return f"(此处有一张图片，名称为：{img_key})"
            else:
                return f"(图片丢失: {rel_path})"

        content = re.sub(r"!\[(.*?)\]\((.*?)\)", replace_image, content)
        
        # 2. 切分文档 (按标题)
        # 简单策略：按 # 标题切分
        lines = content.split('\n')
        current_chunk = []
        current_title = "导言"
        
        chunks_text = []
        
        for line in lines:
            if line.strip().startswith('#'):
                if current_chunk:
                    chunks_text.append((current_title, "\n".join(current_chunk)))
                current_title = line.strip().lstrip('#').strip()
                current_chunk = [line] # 标题也保留在正文里
            else:
                current_chunk.append(line)
                
        if current_chunk:
            chunks_text.append((current_title, "\n".join(current_chunk)))
            
        # 3. 向量化并存储
        for title, text in chunks_text:
            if not text.strip():
                continue
                
            # 限制长度，避免太长
            # 如果太长可以再次切分，这里先简化
            embedding = await self.embedding_client.get_embedding(text)
            if embedding:
                self.chunks.append({
                    "title": title,
                    "text": text,
                    "source": file_path.name,
                    "embedding": embedding
                })

    def _save_index(self):
        data = {
            "chunks": self.chunks,
            "image_map": self.image_map
        }
        self.index_path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    async def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """搜索相关片段"""
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
            
        # 排序
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        
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
