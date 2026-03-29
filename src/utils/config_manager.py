import os
import json
import yaml
from typing import Any, Dict, Optional
from dataclasses import dataclass, asdict

@dataclass
class ModelConfig:
    """模型配置"""
    hidden_dim: int = 256
    num_heads: int = 8
    num_encoder_layers: int = 3
    num_decoder_layers: int = 3
    dropout: float = 0.1
    history_len: int = 5
    pred_len: int = 15
    bev_size: tuple = (500, 500)
    bev_channels: int = 1
    num_modes: int = 3

@dataclass
class DetectionConfig:
    """检测配置"""
    model_name: str = 'yolov8n'
    confidence_threshold: float = 0.25
    iou_threshold: float = 0.45
    max_detections: int = 100

@dataclass
class AdviceConfig:
    """驾驶建议配置"""
    use_llm: bool = True
    llm_api_url: str = 'http://localhost:8000/generate'
    language: str = 'zh'
    max_retries: int = 3
    timeout: int = 30

@dataclass
class VisualizationConfig:
    """可视化配置"""
    trajectory_color: tuple = (0, 255, 0)
    heading_color: tuple = (0, 0, 255)
    point_size: int = 3
    line_width: int = 2
    heading_length: int = 20
    font_scale: float = 0.5
    gradient_color: bool = True
    show_confidence: bool = True
    show_time_steps: bool = True

@dataclass
class VideoConfig:
    """视频配置"""
    fps: int = 10
    video_codec: str = 'MJPG'
    video_extension: str = 'avi'

@dataclass
class PerformanceConfig:
    """性能配置"""
    use_quantization: bool = True
    use_warmup: bool = True
    max_workers: int = 4
    batch_size: int = 1
    memory_limit: float = 0.8

class ConfigManager:
    """配置管理器"""
    
    DEFAULT_CONFIG = {
        'model': ModelConfig(),
        'detection': DetectionConfig(),
        'advice': AdviceConfig(),
        'visualization': VisualizationConfig(),
        'video': VideoConfig(),
        'performance': PerformanceConfig()
    }
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """
        加载配置
        
        Returns:
            配置字典
        """
        if self.config_path and os.path.exists(self.config_path):
            ext = os.path.splitext(self.config_path)[1].lower()
            
            if ext in ['.yaml', '.yml']:
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = yaml.safe_load(f)
            elif ext == '.json':
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    loaded_config = json.load(f)
            else:
                loaded_config = {}
            
            return self._merge_configs(self.DEFAULT_CONFIG, loaded_config)
        else:
            return self.DEFAULT_CONFIG
    
    def _merge_configs(self, default: Dict, loaded: Dict) -> Dict:
        """
        合并配置
        
        Args:
            default: 默认配置
            loaded: 加载的配置
            
        Returns:
            合并后的配置
        """
        result = default.copy()
        
        for key, value in loaded.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值
        
        Args:
            key: 配置键（支持点分隔，如 'model.hidden_dim'）
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            elif hasattr(value, k):
                value = getattr(value, k)
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """
        设置配置值
        
        Args:
            key: 配置键（支持点分隔，如 'model.hidden_dim'）
            value: 配置值
        """
        keys = key.split('.')
        config = self.config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        config[keys[-1]] = value
    
    def save(self, output_path: Optional[str] = None):
        """
        保存配置
        
        Args:
            output_path: 输出路径
        """
        path = output_path or self.config_path
        if not path:
            raise ValueError("No output path specified")
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        ext = os.path.splitext(path)[1].lower()
        
        if ext in ['.yaml', '.yml']:
            with open(path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, allow_unicode=True, default_flow_style=False)
        elif ext == '.json':
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(self.config, f, ensure_ascii=False, indent=2)
        else:
            raise ValueError(f"Unsupported file format: {ext}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        转换为字典
        
        Returns:
            配置字典
        """
        def dataclass_to_dict(obj):
            if isinstance(obj, dict):
                return {k: dataclass_to_dict(v) for k, v in obj.items()}
            elif hasattr(obj, '__dataclass_fields__'):
                return dataclass_to_dict(asdict(obj))
            else:
                return obj
        
        return dataclass_to_dict(self.config)

# 单元测试
if __name__ == "__main__":
    config_manager = ConfigManager()
    
    print("=== Config Test ===")
    print(f"Model hidden_dim: {config_manager.get('model.hidden_dim')}")
    print(f"Detection confidence: {config_manager.get('detection.confidence_threshold')}")
    print(f"Video fps: {config_manager.get('video.fps')}")
    
    config_manager.set('model.hidden_dim', 512)
    print(f"Updated hidden_dim: {config_manager.get('model.hidden_dim')}")
