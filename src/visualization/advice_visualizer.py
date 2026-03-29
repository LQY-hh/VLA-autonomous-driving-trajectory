import cv2
import numpy as np
from typing import Dict, Optional

class AdviceVisualizer:
    """驾驶建议可视化模块"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化驾驶建议可视化器
        
        Args:
            config: 可视化配置参数
        """
        # 默认配置
        default_config = {
            'text_color': (255, 255, 0),         # 文本颜色（黄色）
            'bg_color': (0, 0, 0),               # 背景颜色（黑色）
            'border_color': (255, 255, 0),       # 边框颜色（黄色）
            'font_scale': 0.8,                   # 字体大小
            'font_thickness': 2,                 # 字体粗细
            'border_thickness': 2,               # 边框粗细
            'padding': 15,                       # 内边距
            'margin': 20,                        # 外边距
            'line_spacing': 20                   # 行间距
        }
        
        # 更新配置
        self.config = default_config
        if config:
            self.config.update(config)
        
        self.text_color = self.config['text_color']
        self.bg_color = self.config['bg_color']
        self.border_color = self.config['border_color']
        self.font_scale = self.config['font_scale']
        self.font_thickness = self.config['font_thickness']
        self.border_thickness = self.config['border_thickness']
        self.padding = self.config['padding']
        self.margin = self.config['margin']
        self.line_spacing = self.config['line_spacing']
    
    def visualize_advice(self, image: np.ndarray, advice: str, 
                        confidence: float = 1.0) -> np.ndarray:
        """
        可视化驾驶建议
        
        Args:
            image: 原始图像
            advice: 驾驶建议文本
            confidence: 建议置信度
            
        Returns:
            带有驾驶建议的图像
        """
        # 创建图像副本
        visualized_image = image.copy()
        
        # 分割建议文本为多行
        lines = self._split_text(advice, image.shape[1] - 2 * self.margin - 2 * self.padding)
        
        # 计算文本框大小
        text_size = cv2.getTextSize('Test', cv2.FONT_HERSHEY_SIMPLEX, 
                                 self.font_scale, self.font_thickness)[0]
        line_height = text_size[1] + self.line_spacing
        total_height = len(lines) * line_height + 2 * self.padding
        max_line_width = max([cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 
                                           self.font_scale, self.font_thickness)[0][0] 
                             for line in lines])
        total_width = max_line_width + 2 * self.padding
        
        # 计算文本框位置（右上角）
        x = image.shape[1] - total_width - self.margin
        y = self.margin
        
        # 绘制背景（使用黄色）
        cv2.rectangle(visualized_image, (x, y), 
                     (x + total_width, y + total_height),
                     (255, 255, 0), -1)
        
        # 绘制边框（使用黄色）
        cv2.rectangle(visualized_image, (x, y), 
                     (x + total_width, y + total_height),
                     (255, 255, 0), self.border_thickness)
        
        # 绘制文本（使用黑色）
        for i, line in enumerate(lines):
            text_x = x + self.padding
            text_y = y + self.padding + (i + 1) * line_height - self.line_spacing // 2
            cv2.putText(visualized_image, line, (text_x, text_y),
                       cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                       (0, 0, 0), self.font_thickness)
        
        # 绘制置信度（使用黑色）
        confidence_text = f"Confidence: {confidence:.2f}"
        confidence_size = cv2.getTextSize(confidence_text, cv2.FONT_HERSHEY_SIMPLEX,
                                         self.font_scale * 0.8, self.font_thickness)[0]
        cv2.putText(visualized_image, confidence_text,
                   (x, y + total_height + confidence_size[1] + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.8,
                   (0, 0, 0), self.font_thickness)
        
        return visualized_image
    
    def visualize_advice_with_trajectory(self, image: np.ndarray, advice: str, 
                                       trajectory: list, heading: list, 
                                       confidence: float = 1.0) -> np.ndarray:
        """
        同时可视化驾驶建议和轨迹
        
        Args:
            image: 原始图像
            advice: 驾驶建议文本
            trajectory: 预测轨迹
            heading: 航向信息
            confidence: 建议置信度
            
        Returns:
            带有驾驶建议和轨迹的图像
        """
        # 首先可视化轨迹
        from .trajectory_visualizer import TrajectoryVisualizer
        trajectory_visualizer = TrajectoryVisualizer()
        visualized_image = trajectory_visualizer.visualize_trajectory(
            image, trajectory, heading, confidence
        )
        
        # 然后添加驾驶建议
        visualized_image = self.visualize_advice(visualized_image, advice, confidence)
        
        return visualized_image
    
    def _split_text(self, text: str, max_width: int) -> list:
        """
        分割文本为多行
        
        Args:
            text: 原始文本
            max_width: 最大行宽
            
        Returns:
            分割后的文本行列表
        """
        words = text.split(' ')
        lines = []
        current_line = []
        
        for word in words:
            # 检查当前行加上新单词是否超过最大宽度
            test_line = ' '.join(current_line + [word])
            text_size = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX,
                                     self.font_scale, self.font_thickness)[0]
            
            if text_size[0] <= max_width:
                current_line.append(word)
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                current_line = [word]
        
        # 添加最后一行
        if current_line:
            lines.append(' '.join(current_line))
        
        return lines

# 单元测试
if __name__ == "__main__":
    import os
    
    # 初始化可视化器
    advice_visualizer = AdviceVisualizer()
    
    # 测试图像路径
    test_image_path = "../../data/images/test.jpg"  # 请替换为实际测试图像路径
    
    if os.path.exists(test_image_path):
        print("Testing advice visualization...")
        
        # 加载图像
        image = cv2.imread(test_image_path)
        
        # 测试建议
        test_advice = "保持直线行驶；前方有车辆，保持安全距离；适当减速"
        
        # 可视化建议
        visualized_image = advice_visualizer.visualize_advice(image, test_advice, 0.95)
        
        # 保存可视化结果
        output_path = "../../output/advice_test.jpg"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, visualized_image)
        print(f"Advice visualization saved to {output_path}")
        
        # 测试同时可视化建议和轨迹
        print("\nTesting advice with trajectory visualization...")
        
        # 生成测试轨迹
        height, width, _ = image.shape
        center_x, center_y = width // 2, height // 2
        trajectory = [[center_x + i * 10, center_y + i * 5] for i in range(10)]
        heading = [0.1 * i for i in range(10)]
        
        # 可视化建议和轨迹
        combined_visualized = advice_visualizer.visualize_advice_with_trajectory(
            image, test_advice, trajectory, heading, 0.95
        )
        
        combined_output_path = "../../output/advice_with_trajectory_test.jpg"
        cv2.imwrite(combined_output_path, combined_visualized)
        print(f"Advice with trajectory visualization saved to {combined_output_path}")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Please create a test image in the data/images directory.")
