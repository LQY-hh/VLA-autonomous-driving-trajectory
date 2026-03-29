import cv2
import numpy as np
from typing import Dict, Optional, Tuple, List

class ImagePreprocessor:
    """图像预处理模块"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化图像预处理器
        
        Args:
            config: 预处理配置参数
        """
        # 默认配置
        default_config = {
            'input_size': (640, 640),  # YOLOv8默认输入尺寸
            'normalize': True,
            'mean': [0.485, 0.456, 0.406],  # ImageNet均值
            'std': [0.229, 0.224, 0.225],   # ImageNet标准差
            'color_space': 'RGB'  # 输出色彩空间
        }
        
        # 更新配置
        self.config = default_config
        if config:
            self.config.update(config)
        
        self.input_size = self.config['input_size']
        self.normalize = self.config['normalize']
        self.mean = np.array(self.config['mean'], dtype=np.float32)
        self.std = np.array(self.config['std'], dtype=np.float32)
        self.color_space = self.config['color_space']
    
    def load_image(self, image_path: str) -> np.ndarray:
        """
        加载图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            加载的图像（BGR格式）
        """
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Failed to load image: {image_path}")
        return image
    
    def resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        调整图像尺寸
        
        Args:
            image: 输入图像
            
        Returns:
            调整尺寸后的图像
        """
        resized_image = cv2.resize(image, self.input_size)
        return resized_image
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        归一化图像
        
        Args:
            image: 输入图像
            
        Returns:
            归一化后的图像
        """
        # 将像素值转换为0-1范围
        image = image.astype(np.float32) / 255.0
        
        # 应用均值和标准差
        if self.normalize:
            image = (image - self.mean) / self.std
        
        return image
    
    def convert_color_space(self, image: np.ndarray, src_space: str = 'BGR') -> np.ndarray:
        """
        转换色彩空间
        
        Args:
            image: 输入图像
            src_space: 源色彩空间
            
        Returns:
            转换色彩空间后的图像
        """
        if src_space == 'BGR':
            if self.color_space == 'RGB':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            elif self.color_space == 'GRAY':
                image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        elif src_space == 'RGB':
            if self.color_space == 'BGR':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            elif self.color_space == 'GRAY':
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        return image
    
    def preprocess(self, image_path: str) -> np.ndarray:
        """
        完整预处理流程
        
        Args:
            image_path: 图像路径
            
        Returns:
            预处理后的图像
        """
        # 加载图像
        image = self.load_image(image_path)
        
        # 调整尺寸
        image = self.resize_image(image)
        
        # 转换色彩空间
        image = self.convert_color_space(image, src_space='BGR')
        
        # 归一化
        image = self.normalize_image(image)
        
        # 添加批次维度
        if len(image.shape) == 2:  # 灰度图
            image = np.expand_dims(image, axis=0)  # (H, W) -> (1, H, W)
        else:  # 彩色图
            image = np.transpose(image, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            image = np.expand_dims(image, axis=0)  # (C, H, W) -> (1, C, H, W)
        
        return image
    
    def preprocess_batch(self, image_paths: List[str]) -> np.ndarray:
        """
        批量预处理
        
        Args:
            image_paths: 图像路径列表
            
        Returns:
            预处理后的批量图像
        """
        batch_images = []
        for image_path in image_paths:
            processed_image = self.preprocess(image_path)
            batch_images.append(processed_image)
        
        # 合并批次
        batch_images = np.concatenate(batch_images, axis=0)
        return batch_images

# 单元测试
if __name__ == "__main__":
    import os
    
    preprocessor = ImagePreprocessor()
    
    # 测试图像路径
    test_image_path = "../../data/images/test.jpg"  # 请替换为实际测试图像路径
    
    if os.path.exists(test_image_path):
        print("Testing image preprocessing...")
        
        # 测试完整预处理流程
        processed_image = preprocessor.preprocess(test_image_path)
        print(f"Original image shape: {cv2.imread(test_image_path).shape}")
        print(f"Processed image shape: {processed_image.shape}")
        print(f"Processed image dtype: {processed_image.dtype}")
        print(f"Processed image min/max: {processed_image.min()}, {processed_image.max()}")
        
        # 测试批量预处理
        print("\nTesting batch preprocessing...")
        batch_paths = [test_image_path] * 2
        batch_images = preprocessor.preprocess_batch(batch_paths)
        print(f"Batch shape: {batch_images.shape}")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Please create a test image in the data/images directory.")
