from ultralytics import YOLO
import numpy as np
from typing import Dict, List, Optional, Tuple

class ObjectDetector:
    """目标检测模块，基于YOLOv8"""
    
    def __init__(self, model_path: str = 'yolov8n.pt', config: Optional[Dict] = None):
        """
        初始化目标检测器
        
        Args:
            model_path: 模型权重路径
            config: 检测配置参数
        """
        # 默认配置
        default_config = {
            'conf_threshold': 0.5,  # 置信度阈值
            'iou_threshold': 0.45,   # IOU阈值
            'device': 'cpu',         # 设备（'cpu'或'cuda'）
            'classes': None          # 检测类别，None表示检测所有类别
        }
        
        # 更新配置
        self.config = default_config
        if config:
            self.config.update(config)
        
        self.conf_threshold = self.config['conf_threshold']
        self.iou_threshold = self.config['iou_threshold']
        self.device = self.config['device']
        self.classes = self.config['classes']
        
        # 加载模型
        self.model = YOLO(model_path)
    
    def detect(self, image: np.ndarray) -> List[Dict]:
        """
        执行目标检测
        
        Args:
            image: 输入图像
            
        Returns:
            检测结果列表，每个元素包含检测框、类别和置信度
        """
        # 执行检测
        results = self.model(image, 
                           conf=self.conf_threshold, 
                           iou=self.iou_threshold, 
                           device=self.device, 
                           classes=self.classes)
        
        # 处理检测结果
        detections = []
        for result in results:
            for box in result.boxes:
                # 获取检测框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # 获取置信度和类别
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # 获取类别名称
                class_name = result.names[cls]
                
                # 构建检测结果
                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class_id': cls,
                    'class_name': class_name
                }
                detections.append(detection)
        
        return detections
    
    def detect_batch(self, images: np.ndarray) -> List[List[Dict]]:
        """
        批量检测
        
        Args:
            images: 批量输入图像
            
        Returns:
            批量检测结果
        """
        # 执行批量检测
        results = self.model(images, 
                           conf=self.conf_threshold, 
                           iou=self.iou_threshold, 
                           device=self.device, 
                           classes=self.classes)
        
        # 处理检测结果
        batch_detections = []
        for result in results:
            detections = []
            for box in result.boxes:
                # 获取检测框坐标
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # 获取置信度和类别
                conf = box.conf[0].cpu().numpy()
                cls = int(box.cls[0].cpu().numpy())
                
                # 获取类别名称
                class_name = result.names[cls]
                
                # 构建检测结果
                detection = {
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class_id': cls,
                    'class_name': class_name
                }
                detections.append(detection)
            batch_detections.append(detections)
        
        return batch_detections
    
    def postprocess(self, detections: List[Dict]) -> List[Dict]:
        """
        后处理检测结果
        
        Args:
            detections: 原始检测结果
            
        Returns:
            后处理后的检测结果
        """
        # 过滤低置信度目标
        filtered_detections = [d for d in detections if d['confidence'] >= self.conf_threshold]
        
        # 按置信度排序
        filtered_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        return filtered_detections
    
    def get_class_names(self) -> Dict[int, str]:
        """
        获取模型支持的类别名称
        
        Returns:
            类别ID到类别名称的映射
        """
        return self.model.names

# 单元测试
if __name__ == "__main__":
    import cv2
    import os
    
    # 初始化检测器
    detector = ObjectDetector()
    
    # 测试图像路径
    test_image_path = "../../data/images/test.jpg"  # 请替换为实际测试图像路径
    
    if os.path.exists(test_image_path):
        print("Testing object detection...")
        
        # 加载图像
        image = cv2.imread(test_image_path)
        
        # 执行检测
        detections = detector.detect(image)
        
        # 后处理
        filtered_detections = detector.postprocess(detections)
        
        print(f"Detected {len(filtered_detections)} objects:")
        for i, detection in enumerate(filtered_detections):
            print(f"  {i+1}. {detection['class_name']}: {detection['confidence']:.2f}")
            print(f"     BBox: {detection['bbox']}")
        
        # 测试批量检测
        print("\nTesting batch detection...")
        batch_images = [image, image]
        batch_detections = detector.detect_batch(batch_images)
        print(f"Batch detection results:")
        for i, detections in enumerate(batch_detections):
            print(f"  Image {i+1}: {len(detections)} objects detected")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Please create a test image in the data/images directory.")
