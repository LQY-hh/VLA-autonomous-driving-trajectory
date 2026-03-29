import os
import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass

@dataclass
class ValidationResult:
    """验证结果"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    metadata: Dict

class InputValidator:
    """输入数据验证器"""
    
    def __init__(self, min_image_size: int = 100, max_image_size: int = 4096,
                 allowed_formats: List[str] = None):
        """
        初始化输入验证器
        
        Args:
            min_image_size: 最小图像尺寸
            max_image_size: 最大图像尺寸
            allowed_formats: 允许的图片格式
        """
        self.min_image_size = min_image_size
        self.max_image_size = max_image_size
        self.allowed_formats = allowed_formats or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    
    def validate_image(self, image_path: str) -> ValidationResult:
        """
        验证单张图片
        
        Args:
            image_path: 图片路径
            
        Returns:
            验证结果
        """
        errors = []
        warnings = []
        metadata = {}
        
        # 检查文件是否存在
        if not os.path.exists(image_path):
            return ValidationResult(
                is_valid=False,
                errors=[f"File does not exist: {image_path}"],
                warnings=[],
                metadata={}
            )
        
        # 检查文件格式
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in self.allowed_formats:
            errors.append(f"Unsupported image format: {ext}")
        
        # 尝试读取图片
        try:
            image = cv2.imread(image_path)
            if image is None:
                errors.append(f"Failed to read image: {image_path}")
                return ValidationResult(
                    is_valid=False,
                    errors=errors,
                    warnings=warnings,
                    metadata={}
                )
        except Exception as e:
            errors.append(f"Error reading image: {e}")
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                metadata={}
            )
        
        # 获取图片元数据
        height, width = image.shape[:2]
        channels = image.shape[2] if len(image.shape) > 2 else 1
        metadata = {
            'width': width,
            'height': height,
            'channels': channels,
            'size_bytes': os.path.getsize(image_path)
        }
        
        # 检查图片尺寸
        if width < self.min_image_size or height < self.min_image_size:
            warnings.append(f"Image size too small: {width}x{height}")
        
        if width > self.max_image_size or height > self.max_image_size:
            errors.append(f"Image size too large: {width}x{height}")
        
        # 检查图片质量
        if self._is_image_too_dark(image):
            warnings.append("Image may be too dark")
        
        if self._is_image_too_blurry(image):
            warnings.append("Image may be too blurry")
        
        return ValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            metadata=metadata
        )
    
    def validate_directory(self, directory_path: str, recursive: bool = True) -> Tuple[List[str], List[str], List[ValidationResult]]:
        """
        验证目录中的所有图片
        
        Args:
            directory_path: 目录路径
            recursive: 是否递归搜索
            
        Returns:
            (valid_images, invalid_images, validation_results)
        """
        valid_images = []
        invalid_images = []
        validation_results = []
        
        if not os.path.exists(directory_path):
            return valid_images, invalid_images, validation_results
        
        for root, dirs, files in os.walk(directory_path):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext not in self.allowed_formats:
                    continue
                
                image_path = os.path.join(root, file)
                result = self.validate_image(image_path)
                validation_results.append(result)
                
                if result.is_valid:
                    valid_images.append(image_path)
                else:
                    invalid_images.append(image_path)
            
            if not recursive:
                break
        
        return valid_images, invalid_images, validation_results
    
    def _is_image_too_dark(self, image: np.ndarray, threshold: float = 30.0) -> bool:
        """
        检查图片是否过暗
        
        Args:
            image: 图片数组
            threshold: 亮度阈值
            
        Returns:
            是否过暗
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        mean_brightness = np.mean(gray)
        return mean_brightness < threshold
    
    def _is_image_too_blurry(self, image: np.ndarray, threshold: float = 100.0) -> bool:
        """
        检查图片是否模糊
        
        Args:
            image: 图片数组
            threshold: 清晰度阈值
            
        Returns:
            是否模糊
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) > 2 else image
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        return variance < threshold
    
    def get_validation_report(self, validation_results: List[ValidationResult]) -> str:
        """
        生成验证报告
        
        Args:
            validation_results: 验证结果列表
            
        Returns:
            报告字符串
        """
        total = len(validation_results)
        valid = sum(1 for r in validation_results if r.is_valid)
        invalid = total - valid
        total_warnings = sum(len(r.warnings) for r in validation_results)
        
        report = f"""
=== Input Validation Report ===
Total images: {total}
Valid images: {valid}
Invalid images: {invalid}
Total warnings: {total_warnings}
"""
        return report

# 单元测试
if __name__ == "__main__":
    validator = InputValidator()
    
    # 测试单个图片验证
    test_image = "data/images/sequence_001/000000.jpg"
    if os.path.exists(test_image):
        result = validator.validate_image(test_image)
        print(f"Validation result for {test_image}:")
        print(f"  Is valid: {result.is_valid}")
        print(f"  Errors: {result.errors}")
        print(f"  Warnings: {result.warnings}")
        print(f"  Metadata: {result.metadata}")
    else:
        print(f"Test image not found: {test_image}")
