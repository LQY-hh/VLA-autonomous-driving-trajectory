import cv2
import numpy as np
import os
from typing import List, Dict, Optional

class VideoGenerator:
    """视频生成模块"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化视频生成器
        
        Args:
            config: 视频生成配置参数
        """
        # 默认配置
        default_config = {
            'fps': 10,                  # 帧率
            'video_codec': 'MJPG',      # 视频编码（Windows兼容）
            'video_extension': 'avi',    # 视频扩展名（Windows兼容）
            'width': 1280,              # 视频宽度
            'height': 720,              # 视频高度
            'text_color': (255, 255, 255),  # 文本颜色
            'text_font': cv2.FONT_HERSHEY_SIMPLEX,  # 文本字体
            'text_scale': 0.8,          # 文本大小
            'text_thickness': 2,        # 文本粗细
            'text_position': (10, 30),   # 文本位置
            'trajectory_color': (0, 255, 0),  # 轨迹颜色
            'trajectory_thickness': 3,   # 轨迹线条粗细
            'trajectory_point_size': 4,  # 轨迹点大小
            'heading_color': (0, 0, 255),  # 航向箭头颜色
            'heading_thickness': 2,      # 航向箭头粗细
            'heading_length': 25,        # 航向箭头长度
            'box_color': (255, 0, 0),    # 检测框颜色
            'box_thickness': 2,          # 检测框粗细
            'lane_color': (0, 255, 255),  # 车道线颜色
            'lane_thickness': 3,         # 车道线粗细
            'background_alpha': 0.7,     # 背景透明度
            'use_bev': False,            # 是否使用BEV视图
            'show_fps': True,            # 是否显示FPS
            'show_frame_counter': True,   # 是否显示帧计数器
            'show_timestamp': True,       # 是否显示时间戳
            'show_trajectory_labels': True,  # 是否显示轨迹标签
            'show_advice_background': True,  # 是否显示建议背景
            'advice_text_scale': 0.9,     # 建议文本大小
            'advice_max_lines': 3,        # 建议最大行数
            'scene_info_position': (10, 80)  # 场景信息位置
        }
        
        # 更新配置
        self.config = default_config
        if config:
            self.config.update(config)
        
        self.fps = self.config['fps']
        self.video_codec = self.config['video_codec']
        self.video_extension = self.config['video_extension']
        self.width = self.config['width']
        self.height = self.config['height']
        self.text_color = self.config['text_color']
        self.text_font = self.config['text_font']
        self.text_scale = self.config['text_scale']
        self.text_thickness = self.config['text_thickness']
        self.text_position = self.config['text_position']
        self.trajectory_color = self.config['trajectory_color']
        self.trajectory_thickness = self.config['trajectory_thickness']
        self.trajectory_point_size = self.config['trajectory_point_size']
        self.heading_color = self.config['heading_color']
        self.heading_thickness = self.config['heading_thickness']
        self.heading_length = self.config['heading_length']
        self.box_color = self.config['box_color']
        self.box_thickness = self.config['box_thickness']
        self.lane_color = self.config['lane_color']
        self.lane_thickness = self.config['lane_thickness']
        self.background_alpha = self.config['background_alpha']
        self.use_bev = self.config['use_bev']
        self.show_fps = self.config['show_fps']
        self.show_frame_counter = self.config['show_frame_counter']
        self.show_timestamp = self.config['show_timestamp']
        self.show_trajectory_labels = self.config['show_trajectory_labels']
        self.show_advice_background = self.config['show_advice_background']
        self.advice_text_scale = self.config['advice_text_scale']
        self.advice_max_lines = self.config['advice_max_lines']
        self.scene_info_position = self.config['scene_info_position']
    
    def generate_video(self, image_paths: List[str], output_path: str, 
                      trajectories: Optional[List[List[List[float]]]] = None,
                      headings: Optional[List[List[float]]] = None,
                      advice: Optional[List[str]] = None,
                      detections: Optional[List[List[Dict]]] = None,
                      lanes: Optional[List[List[List[float]]]] = None,
                      scene_info: Optional[List[Dict]] = None) -> bool:
        """
        生成视频
        
        Args:
            image_paths: 图像路径列表
            output_path: 输出视频路径
            trajectories: 轨迹列表，每个图像对应一个轨迹
            headings: 航向信息列表
            advice: 驾驶建议列表
            detections: 目标检测结果列表
            lanes: 车道线列表
            scene_info: 场景信息列表
            
        Returns:
            是否生成成功
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 获取第一个图像的尺寸
            if not image_paths:
                print("No images provided")
                return False
            
            # 读取第一个图像获取尺寸
            first_image = cv2.imread(image_paths[0])
            if first_image is None:
                print(f"Failed to read image: {image_paths[0]}")
                return False
            
            # 使用配置的尺寸或图像尺寸
            height, width, _ = first_image.shape
            video_width = self.width if self.width else width
            video_height = self.height if self.height else height
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (video_width, video_height))
            
            # 处理每个图像
            for i, image_path in enumerate(image_paths):
                # 读取图像
                image = cv2.imread(image_path)
                if image is None:
                    print(f"Failed to read image: {image_path}")
                    continue
                
                # 调整图像尺寸
                image = cv2.resize(image, (video_width, video_height))
                
                # 添加车道线
                if lanes and i < len(lanes):
                    image = self._draw_lanes(image, lanes[i])
                
                # 添加目标检测框
                if detections and i < len(detections):
                    image = self._draw_detections(image, detections[i])
                
                # 添加轨迹
                if trajectories and i < len(trajectories):
                    trajectory = trajectories[i]
                    heading = headings[i] if headings and i < len(headings) else []
                    image = self._draw_trajectory(image, trajectory, heading)
                
                # 添加驾驶建议
                if advice and i < len(advice):
                    image = self._draw_advice(image, advice[i])
                
                # 添加信息
                current_scene_info = scene_info[i] if scene_info and i < len(scene_info) else None
                image = self._draw_info(image, i+1, len(image_paths), current_scene_info)
                
                # 写入视频
                out.write(image)
            
            # 释放视频写入器
            out.release()
            print(f"Video generated successfully: {output_path}")
            return True
        except Exception as e:
            print(f"Error generating video: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def generate_video_from_visualized_images(self, visualized_image_paths: List[str], 
                                           output_path: str) -> bool:
        """
        从已可视化的图像生成视频
        
        Args:
            visualized_image_paths: 已可视化的图像路径列表
            output_path: 输出视频路径
            
        Returns:
            是否生成成功
        """
        try:
            # 确保输出目录存在
            output_dir = os.path.dirname(output_path)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
            
            if not visualized_image_paths:
                print("No visualized images provided")
                return False
            
            # 读取第一个图像获取尺寸
            first_image = cv2.imread(visualized_image_paths[0])
            if first_image is None:
                print(f"Failed to read image: {visualized_image_paths[0]}")
                return False
            
            height, width, _ = first_image.shape
            
            # 创建视频写入器
            fourcc = cv2.VideoWriter_fourcc(*self.video_codec)
            out = cv2.VideoWriter(output_path, fourcc, self.fps, (width, height))
            
            if not out.isOpened():
                print(f"Failed to open video writer: {output_path}")
                return False
            
            # 处理每个图像
            success_count = 0
            for i, image_path in enumerate(visualized_image_paths):
                try:
                    # 读取图像
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Failed to read image: {image_path}")
                        continue
                    
                    # 调整图像尺寸以匹配视频尺寸
                    if image.shape[0] != height or image.shape[1] != width:
                        image = cv2.resize(image, (width, height))
                    
                    # 添加时间戳
                    timestamp = f"Frame: {i+1}/{len(visualized_image_paths)}"
                    cv2.putText(image, timestamp, self.text_position,
                               self.text_font, self.text_scale,
                               self.text_color, self.text_thickness)
                    
                    # 写入视频
                    out.write(image)
                    success_count += 1
                except Exception as e:
                    print(f"Error processing image {image_path}: {e}")
                    continue
            
            # 释放视频写入器
            out.release()
            
            if success_count == 0:
                print("No images were successfully processed")
                return False
            
            print(f"Video generated successfully: {output_path}")
            print(f"Processed {success_count}/{len(visualized_image_paths)} images")
            return True
        except Exception as e:
            print(f"Error generating video: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _draw_trajectory(self, image: np.ndarray, trajectory: List[List[float]], 
                        heading: List[float]) -> np.ndarray:
        """
        在图像上绘制轨迹
        
        Args:
            image: 输入图像
            trajectory: 轨迹坐标
            heading: 航向信息
            
        Returns:
            带有轨迹的图像
        """
        # 绘制轨迹线
        if len(trajectory) > 1:
            for i in range(len(trajectory) - 1):
                start_point = tuple(map(int, trajectory[i]))
                end_point = tuple(map(int, trajectory[i+1]))
                cv2.line(image, start_point, end_point, self.trajectory_color, self.trajectory_thickness)
        
        # 绘制轨迹点
        for i, point in enumerate(trajectory):
            cv2.circle(image, tuple(map(int, point)), self.trajectory_point_size, self.trajectory_color, -1)
            
            # 绘制轨迹标签
            if self.show_trajectory_labels and i % 2 == 0:  # 每2个点显示一个标签
                label = f"{i*0.5}s"  # 假设每个点间隔0.5秒
                cv2.putText(image, label, (int(point[0]) + 10, int(point[1]) - 10),
                           self.text_font, 0.6, self.text_color, 1)
        
        # 绘制航向箭头
        if heading and len(heading) == len(trajectory):
            for i, (point, heading_angle) in enumerate(zip(trajectory, heading)):
                end_x = point[0] + self.heading_length * np.cos(heading_angle)
                end_y = point[1] + self.heading_length * np.sin(heading_angle)
                cv2.arrowedLine(image, tuple(map(int, point)),
                              tuple(map(int, [end_x, end_y])),
                              self.heading_color, self.heading_thickness)
        
        return image
    
    def _draw_advice(self, image: np.ndarray, advice: str) -> np.ndarray:
        """
        在图像上绘制驾驶建议
        
        Args:
            image: 输入图像
            advice: 驾驶建议文本
            
        Returns:
            带有驾驶建议的图像
        """
        # 分割建议文本为多行
        lines = advice.split('；')
        
        # 限制最大行数
        lines = lines[:self.advice_max_lines]
        
        # 计算文本高度
        text_height = int(35 * len(lines))
        
        # 创建半透明背景
        if self.show_advice_background:
            overlay = image.copy()
            height, width = image.shape[:2]
            cv2.rectangle(overlay, (10, height - text_height - 20), (width - 10, height - 10), (0, 0, 0), -1)
            cv2.addWeighted(overlay, self.background_alpha, image, 1 - self.background_alpha, 0, image)
        
        # 绘制建议
        height, width = image.shape[:2]
        start_y = height - text_height - 10
        for i, line in enumerate(lines):
            text_position = (20, start_y + i * 35)
            cv2.putText(image, line, text_position,
                       self.text_font, self.advice_text_scale,
                       self.text_color, self.text_thickness)
        
        return image
    
    def _draw_detections(self, image: np.ndarray, detections: List[Dict]) -> np.ndarray:
        """
        在图像上绘制目标检测结果
        
        Args:
            image: 输入图像
            detections: 目标检测结果列表
            
        Returns:
            带有检测结果的图像
        """
        for detection in detections:
            # 绘制检测框
            if 'bbox' in detection:
                bbox = detection['bbox']
                x1, y1, x2, y2 = map(int, bbox)
                cv2.rectangle(image, (x1, y1), (x2, y2), self.box_color, self.box_thickness)
                
                # 绘制类别和置信度
                if 'class_name' in detection and 'confidence' in detection:
                    class_name = detection['class_name']
                    confidence = detection['confidence']
                    label = f"{class_name}: {confidence:.2f}"
                    cv2.putText(image, label, (x1, y1 - 10),
                               self.text_font, 0.5, self.text_color, 1)
        
        return image
    
    def _draw_lanes(self, image: np.ndarray, lanes: List[List[float]]) -> np.ndarray:
        """
        在图像上绘制车道线
        
        Args:
            image: 输入图像
            lanes: 车道线坐标列表
            
        Returns:
            带有车道线的图像
        """
        for lane in lanes:
            if len(lane) > 1:
                points = [tuple(map(int, point)) for point in lane]
                cv2.polylines(image, [np.array(points)], False, self.lane_color, self.lane_thickness)
        
        return image
    
    def _draw_info(self, image: np.ndarray, frame_num: int, total_frames: int, 
                   scene_info: Optional[Dict] = None) -> np.ndarray:
        """
        在图像上绘制信息
        
        Args:
            image: 输入图像
            frame_num: 当前帧号
            total_frames: 总帧数
            scene_info: 场景信息
            
        Returns:
            带有信息的图像
        """
        # 创建半透明背景
        overlay = image.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, self.background_alpha, image, 1 - self.background_alpha, 0, image)
        
        # 绘制帧计数器
        if self.show_frame_counter:
            frame_text = f"Frame: {frame_num}/{total_frames}"
            cv2.putText(image, frame_text, (20, 30),
                       self.text_font, self.text_scale,
                       self.text_color, self.text_thickness)
        
        # 绘制FPS
        if self.show_fps:
            fps_text = f"FPS: {self.fps}"
            cv2.putText(image, fps_text, (20, 60),
                       self.text_font, self.text_scale,
                       self.text_color, self.text_thickness)
        
        # 绘制场景信息
        if scene_info:
            info_y = 90
            if 'road_type' in scene_info:
                road_type_text = f"Road: {scene_info['road_type']}"
                cv2.putText(image, road_type_text, (20, info_y),
                           self.text_font, self.text_scale,
                           self.text_color, self.text_thickness)
                info_y += 30
            if 'context' in scene_info and scene_info['context']:
                context = scene_info['context']
                if 'traffic_light' in context:
                    light_text = f"Traffic Light: {context['traffic_light']}"
                    cv2.putText(image, light_text, (20, info_y),
                               self.text_font, self.text_scale,
                               self.text_color, self.text_thickness)
        
        return image

# 单元测试
if __name__ == "__main__":
    import tempfile
    
    # 初始化视频生成器
    video_generator = VideoGenerator()
    
    # 测试图像路径
    test_image_path = "../../data/images/test.jpg"  # 请替换为实际测试图像路径
    
    if os.path.exists(test_image_path):
        print("Testing video generation...")
        
        # 创建临时图像列表
        image_paths = [test_image_path] * 10
        
        # 生成测试轨迹和建议
        trajectories = []
        headings = []
        advice_list = []
        detections = []
        lanes = []
        scene_info_list = []
        
        height, width, _ = cv2.imread(test_image_path).shape
        center_x, center_y = width // 2, height // 2
        
        for i in range(10):
            # 生成轨迹
            trajectory = [[center_x + j * 10, center_y + j * 5] for j in range(5)]
            trajectories.append(trajectory)
            
            # 生成航向
            heading = [0.1 * j for j in range(5)]
            headings.append(heading)
            
            # 生成建议
            advice = f"保持直线行驶；第{i+1}帧"
            advice_list.append(advice)
            
            # 生成检测结果
            detection = [
                {
                    'bbox': [center_x - 50, center_y - 50, center_x + 50, center_y + 50],
                    'class_name': 'car',
                    'confidence': 0.9
                }
            ]
            detections.append(detection)
            
            # 生成车道线
            lane = [[100, center_y], [width - 100, center_y]]
            lanes.append([lane])
            
            # 生成场景信息
            scene_info = {
                'road_type': 'urban',
                'context': {'traffic_light': 'green' if i < 5 else 'yellow'}
            }
            scene_info_list.append(scene_info)
        
        # 生成视频
        output_path = "../../output/test_video.avi"  # 使用avi格式以确保Windows兼容
        success = video_generator.generate_video(
            image_paths, output_path, trajectories, headings, advice_list, detections, lanes, scene_info_list
        )
        
        if success:
            print(f"Test video generated at: {output_path}")
        else:
            print("Failed to generate test video")
        
        # 测试从已可视化图像生成视频
        print("\nTesting video generation from visualized images...")
        # 首先生成一些可视化图像
        visualized_image_paths = []
        for i, image_path in enumerate(image_paths):
            image = cv2.imread(image_path)
            image = cv2.resize(image, (video_generator.width, video_generator.height))
            
            # 绘制轨迹
            if trajectories and i < len(trajectories):
                trajectory = trajectories[i]
                heading = headings[i] if headings and i < len(headings) else []
                image = video_generator._draw_trajectory(image, trajectory, heading)
            
            # 绘制建议
            if advice_list and i < len(advice_list):
                image = video_generator._draw_advice(image, advice_list[i])
            
            # 绘制检测框
            if detections and i < len(detections):
                image = video_generator._draw_detections(image, detections[i])
            
            # 绘制车道线
            if lanes and i < len(lanes):
                image = video_generator._draw_lanes(image, lanes[i])
            
            # 绘制信息
            image = video_generator._draw_info(image, i+1, len(image_paths))
            
            # 保存可视化图像
            visualized_path = f"../../output/visualized_{i}.jpg"
            cv2.imwrite(visualized_path, image)
            visualized_image_paths.append(visualized_path)
        
        # 从可视化图像生成视频
        output_path_from_visualized = "../../output/test_video_from_visualized.avi"
        success = video_generator.generate_video_from_visualized_images(
            visualized_image_paths, output_path_from_visualized
        )
        
        if success:
            print(f"Test video from visualized images generated at: {output_path_from_visualized}")
        else:
            print("Failed to generate test video from visualized images")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Please create a test image in the data/images directory.")
        
        # 创建一个临时测试图像
        print("\nCreating a temporary test image...")
        temp_image = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(temp_image, "Test Image", (500, 360),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        
        # 确保目录存在
        os.makedirs("../../data/images", exist_ok=True)
        cv2.imwrite(test_image_path, temp_image)
        print(f"Created temporary test image at: {test_image_path}")
        
        # 重新运行测试
        print("\nRetrying video generation test...")
        # 初始化视频生成器
        video_generator = VideoGenerator()
        
        # 创建临时图像列表
        image_paths = [test_image_path] * 10
        
        # 生成测试轨迹和建议
        trajectories = []
        headings = []
        advice_list = []
        detections = []
        lanes = []
        scene_info_list = []
        
        height, width, _ = temp_image.shape
        center_x, center_y = width // 2, height // 2
        
        for i in range(10):
            # 生成轨迹
            trajectory = [[center_x + j * 10, center_y + j * 5] for j in range(5)]
            trajectories.append(trajectory)
            
            # 生成航向
            heading = [0.1 * j for j in range(5)]
            headings.append(heading)
            
            # 生成建议
            advice = f"保持直线行驶；第{i+1}帧"
            advice_list.append(advice)
            
            # 生成检测结果
            detection = [
                {
                    'bbox': [center_x - 50, center_y - 50, center_x + 50, center_y + 50],
                    'class_name': 'car',
                    'confidence': 0.9
                }
            ]
            detections.append(detection)
            
            # 生成车道线
            lane = [[100, center_y], [width - 100, center_y]]
            lanes.append([lane])
            
            # 生成场景信息
            scene_info = {
                'road_type': 'urban',
                'context': {'traffic_light': 'green' if i < 5 else 'yellow'}
            }
            scene_info_list.append(scene_info)
        
        # 生成视频
        output_path = "../../output/test_video.avi"  # 使用avi格式以确保Windows兼容
        success = video_generator.generate_video(
            image_paths, output_path, trajectories, headings, advice_list, detections, lanes, scene_info_list
        )
        
        if success:
            print(f"Test video generated at: {output_path}")
        else:
            print("Failed to generate test video")
