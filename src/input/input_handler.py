import os
import glob
import re
import cv2
import numpy as np
import json
from typing import List, Tuple, Optional

class InputHandler:
    """输入处理模块，负责处理摄像头图片文件夹"""
    
    def __init__(self):
        """初始化输入处理器"""
        # 支持的图片格式
        self.supported_formats = ['.jpg', '.jpeg', '.png', '.bmp']
        # 支持的视频格式
        self.supported_video_formats = ['.mp4', '.avi', '.mov', '.mkv']
        # 视频帧提取参数
        self.frame_interval = 5
        # 场景变化阈值
        self.scene_change_threshold = 0.3
        # 图像质量阈值
        self.quality_threshold = 0.5
    
    def scan_directory(self, directory: str, recursive: bool = False) -> List[str]:
        """
        扫描目录，获取所有图片文件路径
        
        Args:
            directory: 目录路径
            recursive: 是否递归扫描子目录
            
        Returns:
            图片文件路径列表
        """
        # 验证路径是否存在
        if not os.path.exists(directory):
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        # 验证路径是否为目录
        if not os.path.isdir(directory):
            raise NotADirectoryError(f"Path is not a directory: {directory}")
        
        # 确保路径是绝对路径
        directory = os.path.abspath(directory)
        
        image_paths = []
        
        try:
            if recursive:
                # 递归扫描所有子目录
                for root, _, files in os.walk(directory):
                    # 预编译扩展名匹配，提高性能
                    for file in files:
                        if any(file.lower().endswith(ext) for ext in self.supported_formats):
                            image_paths.append(os.path.join(root, file))
            else:
                # 只扫描当前目录
                # 使用更高效的方法获取文件列表
                files = os.listdir(directory)
                for file in files:
                    if any(file.lower().endswith(ext) for ext in self.supported_formats):
                        image_paths.append(os.path.join(directory, file))
        except PermissionError:
            raise PermissionError(f"Permission denied: {directory}")
        except Exception as e:
            raise IOError(f"Error reading directory: {e}")
        
        return image_paths
    
    def filter_images(self, file_paths: List[str]) -> List[str]:
        """
        过滤图片文件
        
        Args:
            file_paths: 文件路径列表
            
        Returns:
            过滤后的图片文件路径列表
        """
        # 使用列表推导式提高性能
        return [path for path in file_paths if any(path.lower().endswith(ext) for ext in self.supported_formats)]
    
    def sort_images(self, image_paths: List[str], sort_by: str = 'filename') -> List[str]:
        """
        排序图片文件
        
        Args:
            image_paths: 图片文件路径列表
            sort_by: 排序方式，可选 'filename' 或 'timestamp'
            
        Returns:
            排序后的图片文件路径列表
        """
        if sort_by == 'filename':
            # 智能文件名排序，支持数字序列
            def natural_sort_key(path):
                basename = os.path.basename(path)
                # 提取文件名中的数字部分
                parts = re.split(r'(\d+)', basename)
                # 将数字部分转换为整数，非数字部分保持原样
                return [int(part) if part.isdigit() else part for part in parts]
            return sorted(image_paths, key=natural_sort_key)
        elif sort_by == 'timestamp':
            # 按文件修改时间排序
            return sorted(image_paths, key=lambda x: os.path.getmtime(x))
        else:
            raise ValueError(f"Invalid sort_by value: {sort_by}")
    
    def batch_images(self, image_paths: List[str], batch_size: int = 1) -> List[List[str]]:
        """
        批量处理图片
        
        Args:
            image_paths: 图片文件路径列表
            batch_size: 批量大小
            
        Returns:
            批次列表，每个批次包含指定数量的图片路径
        """
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        batches = []
        for i in range(0, len(image_paths), batch_size):
            batch = image_paths[i:i + batch_size]
            batches.append(batch)
        return batches
    
    def _evaluate_image_quality(self, image: np.ndarray) -> float:
        """
        评估图像质量
        
        Args:
            image: 输入图像
            
        Returns:
            图像质量分数 (0-1)
        """
        # 计算图像清晰度 (Laplacian方差)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 归一化到0-1范围
        quality = min(laplacian_var / 1000.0, 1.0)
        return quality
    
    def _detect_scene_change(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """
        检测场景变化
        
        Args:
            frame1: 前一帧
            frame2: 当前帧
            
        Returns:
            是否发生场景变化
        """
        # 调整大小以提高计算速度
        frame1_resized = cv2.resize(frame1, (320, 240))
        frame2_resized = cv2.resize(frame2, (320, 240))
        
        # 转换为灰度图
        frame1_gray = cv2.cvtColor(frame1_resized, cv2.COLOR_BGR2GRAY)
        frame2_gray = cv2.cvtColor(frame2_resized, cv2.COLOR_BGR2GRAY)
        
        # 计算绝对差异
        diff = cv2.absdiff(frame1_gray, frame2_gray)
        
        # 计算差异均值
        mean_diff = diff.mean() / 255.0
        
        # 判断是否发生场景变化
        return mean_diff > self.scene_change_threshold
    
    def extract_frames_from_video(self, video_path: str) -> List[str]:
        """
        从视频中提取关键帧并保存为图片
        
        Args:
            video_path: 视频文件路径
            
        Returns:
            提取的关键帧图片路径列表
        """
        frame_paths = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Failed to open video: {video_path}")
            return frame_paths
        
        # 创建输出目录
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = os.path.join(os.path.dirname(video_path), f"{video_name}_frames")
        os.makedirs(output_dir, exist_ok=True)
        
        prev_frame = None
        frame_count = 0
        saved_frame_count = 0
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # 按间隔提取帧
                if frame_count % self.frame_interval != 0:
                    frame_count += 1
                    continue
                
                # 评估图像质量
                if self._evaluate_image_quality(frame) < self.quality_threshold:
                    frame_count += 1
                    continue
                
                # 检测场景变化
                if prev_frame is None or self._detect_scene_change(prev_frame, frame):
                    # 保存帧
                    frame_path = os.path.join(output_dir, f"frame_{saved_frame_count:06d}.jpg")
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                    prev_frame = frame
                    saved_frame_count += 1
                
                frame_count += 1
        finally:
            cap.release()
        
        return frame_paths
    
    def add_other_vehicles(self, image: np.ndarray) -> np.ndarray:
        """
        在道路上添加其他车辆
        
        Args:
            image: 输入图像
            
        Returns:
            带有其他车辆的图像
        """
        # 创建图像副本
        image_with_vehicles = image.copy()
        height, width = image_with_vehicles.shape[:2]
        
        # 定义其他车辆的位置和颜色（增大尺寸，调整位置）
        other_vehicles = [
            # (x, y, width, height, color, label)
            (width // 2, height - 80, 40, 20, (0, 255, 0), "车辆1"),  # 绿色车辆
            (width * 2 // 3, height - 60, 45, 22, (255, 0, 0), "车辆2"),  # 蓝色车辆
            (width // 4, height - 100, 35, 18, (0, 255, 255), "车辆3"),  # 青色车辆
            (width * 3 // 4, height - 90, 42, 20, (255, 0, 255), "车辆4")  # 品红色车辆
        ]
        
        # 绘制其他车辆
        for x, y, w, h, color, label in other_vehicles:
            # 绘制车身
            cv2.rectangle(image_with_vehicles, 
                         (x - w//2, y - h), 
                         (x + w//2, y), 
                         color, -1)
            
            # 绘制车轮
            wheel_radius = 4
            cv2.circle(image_with_vehicles, (x - w//3, y), wheel_radius, (0, 0, 0), -1)
            cv2.circle(image_with_vehicles, (x + w//3, y), wheel_radius, (0, 0, 0), -1)
            
            # 添加标签（增大字体）
            cv2.putText(image_with_vehicles, label, 
                       (x - 20, y - h - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, 
                       (255, 255, 255), 2)
        
        return image_with_vehicles
    
    def mark_user_vehicle(self, image: np.ndarray) -> np.ndarray:
        """
        标记用户车辆
        
        Args:
            image: 输入图像
            
        Returns:
            带有用户车辆标记的图像
        """
        # 创建图像副本
        marked_image = image.copy()
        
        # 确保用户车辆永远居中偏左
        height, width = marked_image.shape[:2]
        # 居中偏左位置（宽度的1/3处）
        center_x = width // 3
        bottom_y = height - 40
        
        # 绘制用户车辆标记
        # 绘制一个蓝色的矩形表示用户车辆（增大尺寸）
        car_width = 50
        car_height = 25
        cv2.rectangle(marked_image, 
                     (center_x - car_width//2, bottom_y - car_height), 
                     (center_x + car_width//2, bottom_y), 
                     (0, 0, 255), -1)  # 使用红色，更醒目
        
        # 绘制车轮
        wheel_radius = 5
        cv2.circle(marked_image, (center_x - car_width//3, bottom_y), wheel_radius, (0, 0, 0), -1)
        cv2.circle(marked_image, (center_x + car_width//3, bottom_y), wheel_radius, (0, 0, 0), -1)
        
        # 添加文字标签（增大字体）
        cv2.putText(marked_image, "本车", 
                   (center_x - 20, bottom_y - 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, 
                   (255, 255, 255), 2)
        
        # 添加其他车辆
        marked_image = self.add_other_vehicles(marked_image)
        
        return marked_image
    
    def save_vehicle_annotations(self, image_path: str, user_vehicle_info: dict, other_vehicles_info: list, output_dir: str):
        """
        保存车辆标注信息到文件
        
        Args:
            image_path: 图片路径
            user_vehicle_info: 本车信息
            other_vehicles_info: 其他车辆信息
            output_dir: 输出目录
        """
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 生成标注文件名
        image_name = os.path.basename(image_path)
        annotation_name = os.path.splitext(image_name)[0] + ".json"
        annotation_path = os.path.join(output_dir, annotation_name)
        
        # 构建标注数据
        annotation_data = {
            "image_name": image_name,
            "user_vehicle": user_vehicle_info,
            "other_vehicles": other_vehicles_info,
            "timestamp": os.path.getmtime(image_path)
        }
        
        # 保存标注文件
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(annotation_data, f, ensure_ascii=False, indent=2)
        
        print(f"Saved annotation: {annotation_path}")
    
    def process_input(self, input_path: str, recursive: bool = False, 
                     sort_by: str = 'filename', batch_size: int = 1) -> List[List[str]]:
        """
        处理输入，返回批次列表
        
        Args:
            input_path: 输入路径，可以是目录或视频文件
            recursive: 是否递归扫描子目录
            sort_by: 排序方式
            batch_size: 批量大小
            
        Returns:
            批次列表
        """
        image_paths = []
        
        if os.path.isdir(input_path):
            # 处理目录
            image_paths = self.scan_directory(input_path, recursive)
        elif any(input_path.lower().endswith(ext) for ext in self.supported_video_formats):
            # 处理视频文件
            image_paths = self.extract_frames_from_video(input_path)
        elif any(input_path.lower().endswith(ext) for ext in self.supported_formats):
            # 处理单个图片文件
            image_paths = [input_path]
        else:
            raise ValueError(f"Unsupported input format: {input_path}")
        
        # 过滤图片
        filtered_paths = self.filter_images(image_paths)
        
        # 排序图片
        sorted_paths = self.sort_images(filtered_paths, sort_by)
        
        # 批量处理
        batches = self.batch_images(sorted_paths, batch_size)
        
        return batches

# 单元测试
if __name__ == "__main__":
    handler = InputHandler()
    
    # 测试扫描目录
    print("Testing directory scanning...")
    try:
        test_dir = "../../data/images"
        paths = handler.scan_directory(test_dir, recursive=False)
        print(f"Found {len(paths)} images in {test_dir}")
        for path in paths[:5]:  # 显示前5个路径
            print(f"  - {path}")
    except FileNotFoundError as e:
        print(f"Directory not found: {e}")
    
    # 测试排序
    print("\nTesting image sorting...")
    if 'paths' in locals() and paths:
        sorted_paths = handler.sort_images(paths, sort_by='filename')
        print("Sorted by filename:")
        for path in sorted_paths[:5]:
            print(f"  - {os.path.basename(path)}")
    
    # 测试批处理
    print("\nTesting batching...")
    if 'paths' in locals() and paths:
        batches = handler.batch_images(paths, batch_size=2)
        print(f"Created {len(batches)} batches with batch size 2")
        for i, batch in enumerate(batches[:3]):  # 显示前3个批次
            print(f"Batch {i+1}: {[os.path.basename(p) for p in batch]}")
    
    # 测试处理输入
    print("\nTesting input processing...")
    try:
        test_dir = "../../data/images"
        batches = handler.process_input(test_dir, recursive=False, sort_by='filename', batch_size=2)
        print(f"Processed input directory, created {len(batches)} batches")
        for i, batch in enumerate(batches[:3]):
            print(f"Batch {i+1}: {[os.path.basename(p) for p in batch]}")
    except Exception as e:
        print(f"Error processing input: {e}")
    
    # 测试视频处理（如果有视频文件）
    print("\nTesting video processing...")
    # 检查是否有视频文件
    video_files = []
    for ext in handler.supported_video_formats:
        video_files.extend(glob.glob(f"../../data/**/*{ext}", recursive=True))
    
    if video_files:
        test_video = video_files[0]
        print(f"Testing video: {test_video}")
        try:
            frame_paths = handler.extract_frames_from_video(test_video)
            print(f"Extracted {len(frame_paths)} frames from video")
            if frame_paths:
                print(f"First frame: {frame_paths[0]}")
        except Exception as e:
            print(f"Error processing video: {e}")
    else:
        print("No video files found for testing")
