import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

class TrajectoryVisualizer:
    """轨迹可视化模块"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化轨迹可视化器
        
        Args:
            config: 可视化配置参数
        """
        # 默认配置
        default_config = {
            'trajectory_color': (0, 255, 0),  # 轨迹颜色（绿色）
            'heading_color': (0, 0, 255),     # 航向颜色（红色）
            'point_size': 3,                   # 轨迹点大小
            'line_width': 2,                   # 轨迹线宽度
            'heading_length': 20,              # 航向箭头长度
            'font_scale': 0.5,                 # 字体大小
            'font_color': (255, 255, 255),    # 字体颜色
            'font_thickness': 1,               # 字体粗细
            'prediction_horizon': 3.0,         # 预测时间范围（秒）
            'show_confidence': True,           # 显示置信度
            'show_time_steps': True,           # 显示时间步
            'gradient_color': True,            # 使用渐变颜色
            'advice_box_color': (0, 255, 255), # 建议框颜色 (BGR格式的黄色)
            'advice_text_color': (0, 0, 0),    # 建议文本颜色
            'risk_color_map': {                # 风险等级颜色映射
                'low': (0, 255, 0),
                'medium': (255, 255, 0),
                'high': (255, 0, 0)
            },
            'show_scene_info': True,           # 显示场景信息
            'scene_box_color': (0, 255, 255),  # 场景信息框颜色
            'scene_text_color': (0, 0, 0),     # 场景信息文本颜色
            'show_statistics': True,           # 显示统计信息
            'statistics_box_color': (0, 128, 255),  # 统计信息框颜色
            'statistics_text_color': (255, 255, 255),  # 统计信息文本颜色
            'trajectory_velocity_color': True, # 根据速度显示颜色
            'max_velocity': 30.0,              # 最大速度（km/h）
            'min_velocity': 0.0,               # 最小速度（km/h）
            'show_trajectory_labels': True,     # 显示轨迹标签
            'label_font_scale': 0.4,           # 标签字体大小
            'label_font_color': (255, 255, 255), # 标签字体颜色
            'show_object_ids': True,           # 显示目标ID
            'object_id_font_scale': 0.4,       # 目标ID字体大小
            'object_id_font_color': (255, 255, 255), # 目标ID字体颜色
            'object_box_color': (255, 0, 0),   # 目标框颜色
            'object_box_thickness': 2,         # 目标框厚度
            'show_road_info': True,            # 显示道路信息
            'road_info_box_color': (128, 255, 128), # 道路信息框颜色
            'road_info_text_color': (0, 0, 0)  # 道路信息文本颜色
        }
        
        # 更新配置
        self.config = default_config
        if config:
            self.config.update(config)
        
        self.trajectory_color = self.config['trajectory_color']
        self.heading_color = self.config['heading_color']
        self.point_size = self.config['point_size']
        self.line_width = self.config['line_width']
        self.heading_length = self.config['heading_length']
        self.font_scale = self.config['font_scale']
        self.font_color = self.config['font_color']
        self.font_thickness = self.config['font_thickness']
        self.prediction_horizon = self.config['prediction_horizon']
        self.show_confidence = self.config['show_confidence']
        self.show_time_steps = self.config['show_time_steps']
        self.gradient_color = self.config['gradient_color']
        self.advice_box_color = self.config['advice_box_color']
        self.advice_text_color = self.config['advice_text_color']
        self.risk_color_map = self.config['risk_color_map']
        self.show_scene_info = self.config['show_scene_info']
        self.scene_box_color = self.config['scene_box_color']
        self.scene_text_color = self.config['scene_text_color']
        self.show_statistics = self.config['show_statistics']
        self.statistics_box_color = self.config['statistics_box_color']
        self.statistics_text_color = self.config['statistics_text_color']
        self.trajectory_velocity_color = self.config['trajectory_velocity_color']
        self.max_velocity = self.config['max_velocity']
        self.min_velocity = self.config['min_velocity']
        self.show_trajectory_labels = self.config['show_trajectory_labels']
        self.label_font_scale = self.config['label_font_scale']
        self.label_font_color = self.config['label_font_color']
        self.show_object_ids = self.config['show_object_ids']
        self.object_id_font_scale = self.config['object_id_font_scale']
        self.object_id_font_color = self.config['object_id_font_color']
        self.object_box_color = self.config['object_box_color']
        self.object_box_thickness = self.config['object_box_thickness']
        self.show_road_info = self.config['show_road_info']
        self.road_info_box_color = self.config['road_info_box_color']
        self.road_info_text_color = self.config['road_info_text_color']
    
    def visualize_trajectory(self, image: np.ndarray, trajectory: List[List[float]], 
                           heading: List[float], confidence: float = 1.0, 
                           risk_level: str = 'low', scene_info: Optional[Dict] = None, 
                           velocity: Optional[List[float]] = None, detections: Optional[List[Dict]] = None, 
                           road_info: Optional[Dict] = None, lanes: Optional[List[List[float]]] = None) -> np.ndarray:
        """
        可视化轨迹
        
        Args:
            image: 原始图像
            trajectory: 预测轨迹
            heading: 航向信息
            confidence: 轨迹置信度
            risk_level: 风险等级
            scene_info: 场景信息
            velocity: 速度信息
            detections: 目标检测结果
            road_info: 道路信息
            lanes: 车道线信息
            
        Returns:
            带有轨迹可视化的图像
        """
        # 创建图像副本
        visualized_image = image.copy()
        
        # 绘制目标检测框
        if detections and self.show_object_ids:
            for i, detection in enumerate(detections):
                bbox = detection.get('bbox', [])
                if len(bbox) == 4:
                    x1, y1, x2, y2 = map(int, bbox)
                    # 绘制目标框
                    cv2.rectangle(visualized_image, (x1, y1), (x2, y2), 
                                 self.object_box_color, self.object_box_thickness)
                    # 绘制目标ID
                    class_name = detection.get('class_name', 'Unknown')
                    confidence = detection.get('confidence', 0.0)
                    label = f"{class_name} {confidence:.2f}"
                    cv2.putText(visualized_image, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, self.object_id_font_scale,
                               self.object_id_font_color, self.font_thickness)
        
        # 绘制车道线
        if lanes:
            for lane in lanes:
                if len(lane) > 1:
                    # 确保车道线坐标是整数
                    points = [tuple(map(int, point)) for point in lane]
                    # 绘制车道线
                    cv2.polylines(visualized_image, [np.array(points)], False, (255, 255, 0), 2)
        
        # 转换轨迹坐标到图像坐标
        image_trajectory = self._trajectory_to_image_coords(trajectory, image.shape)
        
        # 绘制轨迹线
        if len(image_trajectory) > 1:
            # 平滑轨迹线
            smoothed_trajectory = self._smooth_trajectory(image_trajectory)
            # 使用渐变颜色，从浅绿色到深绿色，显示轨迹的时间顺序
            for i in range(len(smoothed_trajectory) - 1):
                # 计算颜色渐变
                ratio = i / (len(smoothed_trajectory) - 1)
                color = (0, int(150 + 105 * (1 - ratio)), 0)  # 从深绿到浅绿
                start_point = tuple(map(int, smoothed_trajectory[i]))
                end_point = tuple(map(int, smoothed_trajectory[i+1]))
                # 使用抗锯齿线条，提高视觉质量
                cv2.line(visualized_image, start_point, end_point, 
                        color, self.line_width + 2, cv2.LINE_AA)
        
        # 绘制轨迹点
        if len(image_trajectory) > 0:
            for i, point in enumerate(image_trajectory):
                # 计算颜色渐变
                ratio = i / (len(image_trajectory) - 1)
                color = (0, int(150 + 105 * (1 - ratio)), 0)  # 从深绿到浅绿
                # 轨迹点大小随时间增加，突出显示未来位置
                point_size = self.point_size + int(ratio * 2)
                cv2.circle(visualized_image, tuple(map(int, point)), 
                          point_size, color, -1)
                
                # 只在关键时间点标注时间步，避免混乱
                if self.show_time_steps and (i == 0 or i == len(image_trajectory) - 1 or i % 3 == 0):
                    time_step = i * (self.prediction_horizon / len(trajectory))
                    text = f"{time_step:.1f}s"
                    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness)[0]
                    # 调整文本位置，避免重叠
                    text_x = int(point[0]) + 10
                    text_y = int(point[1]) - 10
                    # 确保文本在图像范围内
                    height, width, _ = image.shape
                    text_x = min(text_x, width - text_size[0] - 10)
                    text_y = max(text_size[1] + 10, text_y)
                    # 添加文本背景，提高可读性
                    cv2.rectangle(visualized_image, (text_x - 2, text_y - text_size[1] - 2),
                                 (text_x + text_size[0] + 2, text_y + 2),
                                 (255, 255, 255), -1)
                    cv2.putText(visualized_image, text, (text_x, text_y),
                               cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                               (0, 0, 0), self.font_thickness, cv2.LINE_AA)
        
        # 绘制航向信息
        if heading and len(heading) == len(image_trajectory):
            for i, (point, heading_angle) in enumerate(zip(image_trajectory, heading)):
                # 只在关键点绘制航向箭头
                if i == len(image_trajectory) - 1:
                    # 计算航向箭头终点
                    end_x = point[0] + self.heading_length * np.cos(heading_angle)
                    end_y = point[1] + self.heading_length * np.sin(heading_angle)
                    
                    # 绘制航向箭头
                    cv2.arrowedLine(visualized_image, tuple(map(int, point)),
                                  tuple(map(int, [end_x, end_y])),
                                  (0, 0, 255), self.line_width)
        
        # 绘制置信度
        if self.show_confidence:
            # 确保文本在图像范围内
            height, width, _ = image.shape
            text = f"Confidence: {confidence:.2f}"
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness)[0]
            text_x = 10
            text_y = 30
            # 确保文本在图像范围内
            text_x = min(text_x, width - text_size[0] - 10)
            text_y = min(text_y, height - text_size[1] - 10)
            # 使用黑色文本，确保清晰可见
            cv2.putText(visualized_image, text,
                       (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                       (0, 0, 0), self.font_thickness, cv2.LINE_AA)
        
        # 绘制风险等级
        risk_color = (0, 0, 0)  # 使用黑色
        height, width, _ = image.shape
        text = f"Risk: {risk_level}"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness)[0]
        text_x = 10
        text_y = 60
        # 确保文本在图像范围内
        text_x = min(text_x, width - text_size[0] - 10)
        text_y = min(text_y, height - text_size[1] - 10)
        cv2.putText(visualized_image, text,
                   (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                   risk_color, self.font_thickness, cv2.LINE_AA)
        
        # 绘制本车标记
        self._draw_user_vehicle(visualized_image)
        
        return visualized_image
    
    def _smooth_trajectory(self, trajectory: List[List[float]], smoothing_factor: float = 0.8) -> List[List[float]]:
        """
        平滑轨迹线
        
        Args:
            trajectory: 原始轨迹点
            smoothing_factor: 平滑因子，值越大轨迹越平滑
            
        Returns:
            平滑后的轨迹点
        """
        if len(trajectory) <= 2:
            return trajectory
        
        smoothed = []
        # 保留第一个点
        smoothed.append(trajectory[0])
        
        # 对中间点进行平滑
        for i in range(1, len(trajectory) - 1):
            x = (1 - smoothing_factor) * trajectory[i][0] + smoothing_factor * (trajectory[i-1][0] + trajectory[i+1][0]) / 2
            y = (1 - smoothing_factor) * trajectory[i][1] + smoothing_factor * (trajectory[i-1][1] + trajectory[i+1][1]) / 2
            smoothed.append([x, y])
        
        # 保留最后一个点
        smoothed.append(trajectory[-1])
        
        return smoothed
    
    def _draw_user_vehicle(self, image: np.ndarray):
        """
        绘制本车标记
        
        Args:
            image: 图像
        """
        height, width, _ = image.shape
        # 在图像底部中央绘制本车标记
        center_x = width // 2
        bottom_y = height - 50
        
        # 绘制一个简单的箭头表示本车行驶方向
        arrow_size = 30
        arrow_points = [
            (center_x, bottom_y),
            (center_x - arrow_size // 2, bottom_y - arrow_size),
            (center_x + arrow_size // 2, bottom_y - arrow_size)
        ]
        # 使用更醒目的颜色
        cv2.fillPoly(image, [np.array(arrow_points)], (0, 0, 255))
        
        # 添加"Vehicle"标签，使用英文避免中文乱码
        text = "Vehicle"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        text_x = center_x - text_size[0] // 2
        text_y = bottom_y - arrow_size - 10
        # 确保文本在图像范围内
        text_y = max(text_size[1] + 10, text_y)
        cv2.putText(image, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    def _get_gradient_color(self, t: float) -> Tuple[int, int, int]:
        """
        获取渐变颜色
        
        Args:
            t: 0-1之间的参数
            
        Returns:
            RGB颜色值
        """
        # 从绿色到蓝色的渐变
        r = int(0)
        g = int(255 * (1 - t))
        b = int(255 * t)
        return (b, g, r)
    
    def _get_velocity_color(self, t: float) -> Tuple[int, int, int]:
        """
        根据速度获取颜色
        
        Args:
            t: 0-1之间的参数（速度归一化值）
            
        Returns:
            RGB颜色值
        """
        # 从绿色（低速）到红色（高速）的渐变
        r = int(255 * t)
        g = int(255 * (1 - t))
        b = int(0)
        return (b, g, r)
    
    def _draw_scene_info(self, image: np.ndarray, scene_info: Dict):
        """
        绘制场景信息
        
        Args:
            image: 图像
            scene_info: 场景信息
        """
        height, width, _ = image.shape
        
        # 准备场景信息文本
        text_lines = []
        if 'road_type' in scene_info:
            text_lines.append(f"Road: {scene_info['road_type']}")
        if 'weather' in scene_info:
            text_lines.append(f"Weather: {scene_info['weather']}")
        if 'time_of_day' in scene_info:
            text_lines.append(f"Time: {scene_info['time_of_day']}")
        if 'traffic_density' in scene_info:
            text_lines.append(f"Traffic: {scene_info['traffic_density']}")
        
        if not text_lines:
            return
        
        # 计算文本框大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        max_line_width = 0
        total_height = 0
        for line in text_lines:
            text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            max_line_width = max(max_line_width, text_size[0])
            total_height += text_size[1] + 5
        
        # 添加边距
        padding = 5
        box_width = max_line_width + 2 * padding
        box_height = total_height + 2 * padding
        
        # 计算文本框位置（左上角）
        box_x = 10
        box_y = 90
        
        # 绘制文本框
        cv2.rectangle(image, (box_x, box_y), 
                     (box_x + box_width, box_y + box_height), 
                     self.scene_box_color, -1)
        
        # 绘制文本
        current_y = box_y + padding
        for line in text_lines:
            text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            text_x = box_x + padding
            cv2.putText(image, line, (text_x, current_y + text_size[1]),
                       font, font_scale, self.scene_text_color, font_thickness)
            current_y += text_size[1] + 5
    
    def _draw_statistics(self, image: np.ndarray, stats: Dict):
        """
        绘制统计信息
        
        Args:
            image: 图像
            stats: 统计信息
        """
        height, width, _ = image.shape
        
        # 准备统计信息文本
        text_lines = []
        for key, value in stats.items():
            text_lines.append(f"{key}: {value}")
        
        if not text_lines:
            return
        
        # 计算文本框大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        max_line_width = 0
        total_height = 0
        for line in text_lines:
            text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            max_line_width = max(max_line_width, text_size[0])
            total_height += text_size[1] + 5
        
        # 添加边距
        padding = 5
        box_width = max_line_width + 2 * padding
        box_height = total_height + 2 * padding
        
        # 计算文本框位置（左下角）
        box_x = 10
        box_y = height - box_height - 10
        
        # 绘制文本框
        cv2.rectangle(image, (box_x, box_y), 
                     (box_x + box_width, box_y + box_height), 
                     self.statistics_box_color, -1)
        
        # 绘制文本
        current_y = box_y + padding
        for line in text_lines:
            text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            text_x = box_x + padding
            cv2.putText(image, line, (text_x, current_y + text_size[1]),
                       font, font_scale, self.statistics_text_color, font_thickness)
            current_y += text_size[1] + 5
    
    def _draw_road_info(self, image: np.ndarray, road_info: Dict):
        """
        绘制道路信息
        
        Args:
            image: 图像
            road_info: 道路信息
        """
        height, width, _ = image.shape
        
        # 准备道路信息文本
        text_lines = []
        if 'speed_limit' in road_info:
            text_lines.append(f"Speed Limit: {road_info['speed_limit']} km/h")
        if 'lane_count' in road_info:
            text_lines.append(f"Lanes: {road_info['lane_count']}")
        if 'road_condition' in road_info:
            text_lines.append(f"Condition: {road_info['road_condition']}")
        if 'road_name' in road_info:
            text_lines.append(f"Road: {road_info['road_name']}")
        
        if not text_lines:
            return
        
        # 计算文本框大小
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        max_line_width = 0
        total_height = 0
        for line in text_lines:
            text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            max_line_width = max(max_line_width, text_size[0])
            total_height += text_size[1] + 5
        
        # 添加边距
        padding = 5
        box_width = max_line_width + 2 * padding
        box_height = total_height + 2 * padding
        
        # 计算文本框位置（右下角）
        box_x = width - box_width - 10
        box_y = height - box_height - 10
        
        # 绘制文本框
        cv2.rectangle(image, (box_x, box_y), 
                     (box_x + box_width, box_y + box_height), 
                     self.road_info_box_color, -1)
        
        # 绘制文本
        current_y = box_y + padding
        for line in text_lines:
            text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            text_x = box_x + padding
            cv2.putText(image, line, (text_x, current_y + text_size[1]),
                       font, font_scale, self.road_info_text_color, font_thickness)
            current_y += text_size[1] + 5
    
    def visualize_multiple_trajectories(self, image: np.ndarray, 
                                      trajectories: List[Dict]) -> np.ndarray:
        """
        可视化多个轨迹
        
        Args:
            image: 原始图像
            trajectories: 轨迹列表，每个轨迹包含轨迹点、航向和置信度
            
        Returns:
            带有多个轨迹可视化的图像
        """
        # 创建图像副本
        visualized_image = image.copy()
        
        # 为每个轨迹分配不同颜色
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
        
        # 绘制每个轨迹
        for i, traj_info in enumerate(trajectories):
            trajectory = traj_info['trajectory']
            heading = traj_info.get('heading', [])
            confidence = traj_info.get('confidence', 1.0)
            
            # 选择颜色
            color = colors[i % len(colors)]
            
            # 转换轨迹坐标到图像坐标
            image_trajectory = self._trajectory_to_image_coords(trajectory, image.shape)
            
            # 绘制轨迹线
            if len(image_trajectory) > 1:
                for j in range(len(image_trajectory) - 1):
                    start_point = tuple(map(int, image_trajectory[j]))
                    end_point = tuple(map(int, image_trajectory[j+1]))
                    cv2.line(visualized_image, start_point, end_point, 
                            color, self.line_width)
            
            # 绘制轨迹点
            for j, point in enumerate(image_trajectory):
                cv2.circle(visualized_image, tuple(map(int, point)), 
                          self.point_size, color, -1)
            
            # 绘制置信度
            cv2.putText(visualized_image, f"Mode {i+1}: {confidence:.2f}",
                       (10, 30 + i * 20), cv2.FONT_HERSHEY_SIMPLEX, self.font_scale,
                       color, self.font_thickness)
        
        return visualized_image
    
    def _trajectory_to_image_coords(self, trajectory: List[List[float]], 
                                   image_shape: Tuple[int, int, int]) -> List[List[float]]:
        """
        将轨迹坐标转换为图像坐标
        
        Args:
            trajectory: 轨迹坐标
            image_shape: 图像形状
            
        Returns:
            转换后的图像坐标
        """
        height, width, _ = image_shape
        
        # 计算轨迹的边界
        if not trajectory:
            return []
        
        # 将轨迹映射到道路区域，调整位置和缩放
        image_trajectory = []
        # 调整轨迹显示位置，使其位于道路中央
        center_x = width // 2
        # 将轨迹垂直位置调整到图像中下部，对应实际道路位置
        center_y = height * 0.65  # 调整到图像65%高度处，更符合驾驶视角
        
        # 固定缩放因子，确保轨迹长度合理
        scale = 100
        
        # 确保时间越长的点显示在越远的地方
        # 基于时间步计算y坐标偏移，确保时间增长时y值减小（向上移动）
        for i, point in enumerate(trajectory):
            # 映射轨迹到道路区域
            x = center_x + point[0] * scale
            # 基于时间步和轨迹点计算y坐标，确保时间越长的点越靠上
            # 使用时间步作为主要因素，确保时间顺序与空间位置一致
            time_factor = i / len(trajectory) if len(trajectory) > 0 else 0
            # 结合轨迹点的y值和时间因子，确保时间越长的点越靠上
            y = center_y - (time_factor * 300 + abs(point[1]) * scale)  
            
            # 确保坐标在图像范围内
            x = max(0, min(width - 1, x))
            y = max(0, min(height - 1, y))
            image_trajectory.append([x, y])
        
        return image_trajectory
    
    def visualize_advice(self, image: np.ndarray, advice: str, confidence: float = 1.0) -> np.ndarray:
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
        
        # 获取图像尺寸
        height, width, _ = image.shape
        
        # 确保advice是字符串类型
        if not isinstance(advice, str):
            advice = str(advice)
        
        # 处理空字符串情况
        if not advice:
            advice = "无驾驶建议"
        
        # 计算文本尺寸
        # 分割文本为多行，确保每行不超过一定长度
        def split_text(text, max_chars=15):
            # 处理中文文本，按字符分割
            if any(ord(c) > 127 for c in text):
                lines = []
                current_line = []
                for char in text:
                    if char == '；':
                        if current_line:
                            lines.append(''.join(current_line))
                            current_line = []
                    elif len(''.join(current_line)) < max_chars:
                        current_line.append(char)
                    else:
                        if current_line:
                            lines.append(''.join(current_line))
                            current_line = [char]
                if current_line:
                    lines.append(''.join(current_line))
                return lines
            else:
                # 英文文本按单词分割
                words = text.split(' ')
                lines = []
                current_line = []
                for word in words:
                    if len(' '.join(current_line + [word])) <= max_chars:
                        current_line.append(word)
                    else:
                        if current_line:
                            lines.append(' '.join(current_line))
                        current_line = [word]
                if current_line:
                    lines.append(' '.join(current_line))
                return lines
        
        text_lines = split_text(advice)
        # 使用支持英文的字体
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.2  # 增大字体大小，使文字更显眼
        font_thickness = 4  # 增大字体粗细，提高可读性
        
        # 计算文本框大小
        max_line_width = 0
        total_height = 0
        for line in text_lines:
            text_size = cv2.getTextSize(line, font, font_scale, font_thickness)[0]
            max_line_width = max(max_line_width, text_size[0])
            total_height += text_size[1] + 15  # 增大行间距，提高可读性
        
        # 添加边距
        padding = 30  # 增大边距，使文本框更醒目
        box_width = max_line_width + 2 * padding
        box_height = total_height + 2 * padding
        
        # 计算文本框位置（右上角）
        box_x = width - box_width - 20  # 调整位置，离边缘更近
        box_y = 20  # 调整位置，离边缘更近
        
        # 确保文本框在图像范围内
        box_x = max(10, box_x)
        box_y = max(10, box_y)
        box_width = min(width - 20, box_width)
        box_height = min(height - 20, box_height)
        
        # 强制使用黄色背景 (255, 255, 0) - BGR格式
        yellow_color = (0, 255, 255)  # 注意：OpenCV使用BGR格式，所以黄色是(0, 255, 255)
        
        # 绘制文本框（带圆角）
        self._draw_rounded_rectangle(visualized_image, (box_x, box_y), 
                                   (box_x + box_width, box_y + box_height), 
                                   yellow_color, -1, 15)  # 使用黄色背景，增大圆角
        
        # 绘制文本
        current_y = box_y + padding
        for line in text_lines:
            text_x = box_x + padding
            text_y = current_y + 30  # 增大行高，提高可读性
            # 确保文本在图像范围内
            text_x = max(box_x + padding, text_x)
            text_y = min(box_y + box_height - padding, text_y)
            # 使用黑色文本，确保与黄色背景形成鲜明对比
            cv2.putText(visualized_image, line, (text_x, text_y),
                       font, font_scale, (0, 0, 0), font_thickness, cv2.LINE_AA)
            current_y += 35  # 增大行间距，提高可读性
        
        # 绘制置信度
        confidence_text = f"Confidence: {confidence:.2f}"
        cv2.putText(visualized_image, confidence_text, 
                   (box_x + padding, box_y + box_height + 30),
                   font, font_scale * 0.8, (0, 0, 0), font_thickness, cv2.LINE_AA)
        
        # 绘制建议类型标签
        type_text = "ADVICE"
        type_size = cv2.getTextSize(type_text, font, font_scale * 0.8, font_thickness)[0]
        type_x = box_x + box_width - type_size[0] - 15
        type_y = box_y - 5
        # 确保标签在图像范围内
        type_x = max(box_x, type_x)
        type_y = max(type_size[1] + 15, type_y)
        cv2.rectangle(visualized_image, (type_x - 10, type_y - type_size[1] - 10),
                     (type_x + type_size[0] + 10, type_y + 10),
                     yellow_color, -1)
        cv2.putText(visualized_image, type_text, (type_x, type_y),
                   font, font_scale * 0.8, (0, 0, 0), font_thickness, cv2.LINE_AA)
        
        return visualized_image
    
    def _draw_rounded_rectangle(self, image: np.ndarray, start: Tuple[int, int], 
                               end: Tuple[int, int], color: Tuple[int, int, int], 
                               thickness: int, radius: int):
        """
        绘制带圆角的矩形
        
        Args:
            image: 图像
            start: 起始坐标
            end: 结束坐标
            color: 颜色
            thickness: 厚度（-1表示填充）
            radius: 圆角半径
        """
        x1, y1 = start
        x2, y2 = end
        
        # 绘制矩形的四个边
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
        
        # 绘制四个圆角
        cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)

# 单元测试
if __name__ == "__main__":
    import os
    
    # 初始化可视化器
    visualizer = TrajectoryVisualizer()
    
    # 测试图像路径
    test_image_path = "../../data/images/test.jpg"  # 请替换为实际测试图像路径
    
    if os.path.exists(test_image_path):
        print("Testing trajectory visualization...")
        
        # 加载图像
        image = cv2.imread(test_image_path)
        
        # 生成测试轨迹
        height, width, _ = image.shape
        center_x, center_y = width // 2, height // 2
        
        trajectory = []
        heading = []
        for i in range(10):
            x = center_x + i * 10
            y = center_y + i * 5
            trajectory.append([x, y])
            heading.append(0.1 * i)  # 模拟航向变化
        
        # 可视化轨迹
        visualized_image = visualizer.visualize_trajectory(image, trajectory, heading, 0.95)
        
        # 保存可视化结果
        output_path = "../../output/trajectory_test.jpg"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, visualized_image)
        print(f"Trajectory visualization saved to {output_path}")
        
        # 测试多轨迹可视化
        print("\nTesting multiple trajectories visualization...")
        trajectories = [
            {
                'trajectory': [[center_x + i * 10, center_y + i * 5] for i in range(10)],
                'heading': [0.1 * i for i in range(10)],
                'confidence': 0.95
            },
            {
                'trajectory': [[center_x + i * 10, center_y - i * 5] for i in range(10)],
                'heading': [-0.1 * i for i in range(10)],
                'confidence': 0.85
            }
        ]
        
        multi_visualized = visualizer.visualize_multiple_trajectories(image, trajectories)
        multi_output_path = "../../output/multi_trajectory_test.jpg"
        cv2.imwrite(multi_output_path, multi_visualized)
        print(f"Multiple trajectories visualization saved to {multi_output_path}")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Please create a test image in the data/images directory.")
