import json
from pathlib import Path
from typing import Dict, Any, Optional
import nonebot
from nonebot import logger

class ConfigManager:
    def __init__(self):
        self.base_path = Path("data/Virtual_friends")
        self.personas_path = self.base_path / "personas.json"
        self.groups_config_path = self.base_path / "groups.json"
        
        self.personas: Dict[str, Any] = {}
        self.groups: Dict[str, Any] = {}
        
        self.default_instance_config: Dict[str, Any] = {
            "group_name": "未知群组",
            "persona_name": "default",
            "reply_rate": 0.3,
            "active_mode": False,
            "active_hours": [8, 23],
            "active_check_interval": 45,  # 主动行为检查间隔(分钟)
            "idle_trigger_probability": 0.05,  # 闲聊触发概率
            "silence_threshold": 24
        }
        
    async def initialize(self):
        """初始化配置目录和默认文件"""
        logger.info("初始化配置管理器...")
        self.base_path.mkdir(parents=True, exist_ok=True)
        
        # 创建默认人设库
        if not self.personas_path.exists():
            logger.info("创建默认人设库...")
            default_personas = {
                "default": {
                    "prompt": "你是一个友好、乐于助人的AI助手。",
                    "description": "默认助手"
                }
            }
            self.personas_path.write_text(
                json.dumps(default_personas, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        
        self.personas = json.loads(self.personas_path.read_text(encoding="utf-8"))
        logger.success(f"加载了 {len(self.personas)} 个人设: {', '.join(self.personas.keys())}")
        
        # 加载群组配置
        if not self.groups_config_path.exists():
            logger.info("创建群组配置文件...")
            self.groups_config_path.write_text(
                json.dumps({}, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        
        self.groups = json.loads(self.groups_config_path.read_text(encoding="utf-8"))
        logger.success(f"加载了 {len(self.groups)} 个群组配置")
    
    def _save_groups(self):
        """保存群组配置到文件"""
        # 确保 group_name 在第一位
        ordered_groups = {}
        for gid, config in self.groups.items():
            ordered_config = {}
            if "group_name" in config:
                ordered_config["group_name"] = config["group_name"]
            for k, v in config.items():
                if k != "group_name":
                    ordered_config[k] = v
            ordered_groups[gid] = ordered_config
            
        self.groups_config_path.write_text(
            json.dumps(ordered_groups, ensure_ascii=False, indent=2),
            encoding="utf-8"
        )

    def is_in_whitelist(self, group_id: str) -> bool:
        """检查群组是否在白名单中"""
        return self.groups.get(group_id, {}).get("whitelisted", False)
    
    def add_to_whitelist(self, group_id: str, group_name: str = "未知群组") -> bool:
        """添加群组到白名单"""
        try:
            if group_id not in self.groups:
                self.groups[group_id] = self.default_instance_config.copy()
            
            if self.groups[group_id].get("whitelisted"):
                logger.warning(f"群组 {group_id} 已在白名单中")
                # 更新群名
                if group_name != "未知群组":
                    self.groups[group_id]["group_name"] = group_name
                    self._save_groups()
                return False
            
            self.groups[group_id]["whitelisted"] = True
            self.groups[group_id]["group_name"] = group_name
            self._save_groups()
            
            logger.success(f"已将群组 {group_id} ({group_name}) 添加到白名单")
            return True
        except Exception as e:
            logger.error(f"添加白名单失败: {e}")
            return False
    
    def remove_from_whitelist(self, group_id: str) -> bool:
        """从白名单移除群组"""
        try:
            if not self.is_in_whitelist(group_id):
                logger.warning(f"群组 {group_id} 不在白名单中")
                return False
            
            self.groups[group_id]["whitelisted"] = False
            self._save_groups()
            
            logger.success(f"已将群组 {group_id} 从白名单移除")
            return True
        except Exception as e:
            logger.error(f"移除白名单失败: {e}")
            return False
    
    def get_whitelist(self) -> list:
        """获取白名单列表"""
        return [gid for gid, cfg in self.groups.items() if cfg.get("whitelisted")]
    
    def get_personas(self) -> Dict[str, Any]:
        """获取所有人设"""
        return self.personas
    
    def get_instance_config(self, group_id: str) -> Dict[str, Any]:
        """获取实例配置"""
        if group_id not in self.groups:
            return self.default_instance_config.copy()
        return self.groups[group_id]
    
    def update_instance_config(self, group_id: str, updates: Dict[str, Any]):
        """更新实例配置"""
        logger.info(f"更新群组 {group_id} 配置: {updates}")
        if group_id not in self.groups:
            self.groups[group_id] = self.default_instance_config.copy()
            
        self.groups[group_id].update(updates)
        self._save_groups()
    
    def get_all_group_ids(self) -> list:
        """获取所有已有配置的群组 ID"""
        return list(self.groups.keys())
        


    
    def get_persona_prompt(self, persona_name: str) -> str:
        """获取人设提示词"""
        return self.personas.get(persona_name, self.personas["default"])["prompt"]
    
    def add_persona(self, name: str, prompt: str, description: str) -> bool:
        """添加新人设"""
        try:
            self.personas[name] = {
                "prompt": prompt,
                "description": description
            }
            
            # 保存到文件
            self.personas_path.write_text(
                json.dumps(self.personas, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            
            logger.success(f"添加人设 '{name}': {description}")
            return True
        except Exception as e:
            logger.error(f"添加人设失败: {e}")
            return False
    
    def delete_persona(self, name: str) -> bool:
        """删除人设"""
        try:
            if name not in self.personas:
                logger.warning(f"尝试删除不存在的人设: {name}")
                return False
            
            del self.personas[name]
            
            # 保存到文件
            self.personas_path.write_text(
                json.dumps(self.personas, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            
            logger.success(f"删除人设 '{name}'")
            return True
        except Exception as e:
            logger.error(f"删除人设失败: {e}")
            return False
    
    def update_persona(self, name: str, prompt: Optional[str] = None, description: Optional[str] = None) -> bool:
        """更新人设"""
        try:
            if name not in self.personas:
                logger.warning(f"尝试更新不存在的人设: {name}")
                return False
            
            if prompt is not None:
                self.personas[name]["prompt"] = prompt
            
            if description is not None:
                self.personas[name]["description"] = description
            
            # 保存到文件
            self.personas_path.write_text(
                json.dumps(self.personas, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
            
            logger.success(f"更新人设 '{name}'")
            return True
        except Exception as e:
            logger.error(f"更新人设失败: {e}")
            return False
    
    @staticmethod
    def get_env(key: str, default: str = "") -> str:
        """获取环境变量（支持大小写不敏感）"""
        config = nonebot.get_driver().config
        
        # 尝试原始键名
        value = getattr(config, key, None)
        
        # 如果未找到，尝试小写键名
        if value is None:
            value = getattr(config, key.lower(), None)
        
        # 如果还是未找到，尝试大写键名
        if value is None:
            value = getattr(config, key.upper(), None)
        
        # 转换为字符串
        result = str(value) if value is not None else default
        
        if not result or result == "None":
            logger.warning(f"环境变量 {key} 未设置或为空（已尝试: {key}, {key.lower()}, {key.upper()}）")
            return default
        
        return result
