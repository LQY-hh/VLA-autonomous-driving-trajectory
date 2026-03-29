import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple

class BEVGenerator:
    """BEV（鸟瞰图）生成模块"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化BEV生成器
        
        Args:
            config: BEV配置参数
        """
        # 默认配置
        default_config = {
            # 相机内参
            'camera_matrix': np.array([[900, 0, 640],
                                     [0, 900, 360],
                                     [0, 0, 1]]),
            # 相机外参（旋转和平移）
            'rotation_matrix': np.eye(3),
            'translation_vector': np.array([0, 0, 1.5]),  # 相机高度1.5m
            # BEV参数
            'bev_size': (500, 500),  # BEV图像尺寸
            'bev_resolution': 0.1,    # 每个像素代表的实际距离（米）
            'bev_range': 25,          # BEV范围（米）
            # 感兴趣区域
            'roi': {
                'x_min': 0.2,
                'x_max': 0.8,
                'y_min': 0.5,
                'y_max': 1.0
            }
        }
        
        # 更新配置
        self.config = default_config
        if config:
            self.config.update(config)
        
        self.camera_matrix = self.config['camera_matrix']
        self.rotation_matrix = self.config['rotation_matrix']
        self.translation_vector = self.config['translation_vector']
        self.bev_size = self.config['bev_size']
        self.bev_resolution = self.config['bev_resolution']
        self.bev_range = self.config['bev_range']
        self.roi = self.config['roi']
        
        # 计算多摄像头的透视变换矩阵
        self.homography_matrices = {
            'front': self._compute_homography(),
            'rear': self._compute_homography(rotation=np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]),
                                             translation=np.array([0, 0, 1.5])),
            'left': self._compute_homography(rotation=np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]]),
                                             translation=np.array([-1, 0, 1.5])),
            'right': self._compute_homography(rotation=np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]]),
                                              translation=np.array([1, 0, 1.5]))
        }
        
        # 计算前视摄像头的透视变换矩阵（保持向后兼容）
        self.homography_matrix = self.homography_matrices['front']
    
    def _compute_homography(self, rotation=None, translation=None) -> np.ndarray:
        """
        计算透视变换矩阵
        
        Args:
            rotation: 旋转矩阵，默认为None，使用默认旋转矩阵
            translation: 平移向量，默认为None，使用默认平移向量
            
        Returns:
            透视变换矩阵
        """
        # 使用提供的旋转和平移，或者使用默认值
        rot = rotation if rotation is not None else self.rotation_matrix
        trans = translation if translation is not None else self.translation_vector
        
        # 定义地面上的点（真实世界坐标）
        # 假设车辆前方25米，左右各12.5米的区域
        world_points = np.array([
            [-12.5, 0, 0],      # 左下角
            [12.5, 0, 0],       # 右下角
            [12.5, 25, 0],      # 右上角
            [-12.5, 25, 0]       # 左上角
        ], dtype=np.float32)
        
        # 将世界坐标转换为相机坐标
        camera_points = []
        for point in world_points:
            # 应用旋转和平移
            camera_point = rot @ point + trans
            # 应用相机内参
            image_point = self.camera_matrix @ camera_point
            image_point /= image_point[2]
            camera_points.append(image_point[:2])
        
        camera_points = np.array(camera_points, dtype=np.float32)
        
        # 定义BEV图像上的对应点
        bev_points = np.array([
            [0, self.bev_size[1]],         # 左下角
            [self.bev_size[0], self.bev_size[1]],  # 右下角
            [self.bev_size[0], 0],         # 右上角
            [0, 0]                         # 左上角
        ], dtype=np.float32)
        
        # 计算透视变换矩阵
        homography_matrix, _ = cv2.findHomography(camera_points, bev_points)
        return homography_matrix
    
    def generate_bev(self, image: np.ndarray, detections: List[Dict]) -> Tuple[np.ndarray, Dict]:
        """
        生成BEV特征图
        
        Args:
            image: 输入图像
            detections: 目标检测结果
            
        Returns:
            BEV特征图和BEV中的目标信息
        """
        # 创建BEV特征图
        bev_map = np.zeros(self.bev_size, dtype=np.float32)
        
        # 处理检测结果
        bev_objects = []
        for detection in detections:
            # 获取检测框
            x1, y1, x2, y2 = detection['bbox']
            class_name = detection['class_name']
            
            # 计算检测框底部中心（假设目标在地面上）
            bottom_center = np.array([(x1 + x2) / 2, y2], dtype=np.float32)
            
            # 转换到BEV坐标
            bev_point = self._image_to_bev(bottom_center)
            
            if bev_point is not None:
                # 在BEV图上标记目标
                self._mark_object(bev_map, bev_point, class_name)
                
                # 保存BEV中的目标信息
                bev_object = {
                    'class_name': class_name,
                    'bev_coordinates': bev_point.tolist(),
                    'confidence': detection['confidence']
                }
                bev_objects.append(bev_object)
        
        # 生成道路结构
        self._generate_road_structure(bev_map)
        
        return bev_map, bev_objects
    
    def _image_to_bev(self, point: np.ndarray) -> Optional[np.ndarray]:
        """
        将图像坐标转换为BEV坐标
        
        Args:
            point: 图像坐标
            
        Returns:
            BEV坐标，如果点在BEV范围内
        """
        # 添加齐次坐标
        point_hom = np.array([point[0], point[1], 1], dtype=np.float32)
        
        # 应用透视变换
        bev_point_hom = self.homography_matrix @ point_hom
        bev_point_hom /= bev_point_hom[2]
        
        # 检查点是否在BEV范围内
        if (0 <= bev_point_hom[0] < self.bev_size[0] and 
            0 <= bev_point_hom[1] < self.bev_size[1]):
            return bev_point_hom[:2]
        else:
            return None
    
    def _mark_object(self, bev_map: np.ndarray, point: np.ndarray, class_name: str):
        """
        在BEV图上标记目标
        
        Args:
            bev_map: BEV特征图
            point: BEV坐标
            class_name: 目标类别
        """
        # 根据目标类别设置不同的标记值
        if class_name == 'car':
            value = 0.8
        elif class_name == 'pedestrian':
            value = 0.6
        elif class_name == 'bicycle':
            value = 0.4
        else:
            value = 0.2
        
        # 在BEV图上绘制圆形标记
        cv2.circle(bev_map, (int(point[0]), int(point[1])), 5, value, -1)
    
    def _generate_road_structure(self, bev_map: np.ndarray):
        """
        生成道路结构
        
        Args:
            bev_map: BEV特征图
        """
        # 绘制道路中心线
        center_x = self.bev_size[0] // 2
        cv2.line(bev_map, (center_x, 0), (center_x, self.bev_size[1]), 0.5, 2)
        
        # 绘制车道线
        lane_width = 3  # 车道宽度（米）
        lane_pixels = int(lane_width / self.bev_resolution)
        
        # 左侧车道线
        cv2.line(bev_map, (center_x - lane_pixels, 0), (center_x - lane_pixels, self.bev_size[1]), 0.3, 1)
        # 右侧车道线
        cv2.line(bev_map, (center_x + lane_pixels, 0), (center_x + lane_pixels, self.bev_size[1]), 0.3, 1)
    
    def visualize_bev(self, bev_map: np.ndarray) -> np.ndarray:
        """
        可视化BEV图
        
        Args:
            bev_map: BEV特征图
            
        Returns:
            可视化后的BEV图像
        """
        # 归一化到0-255范围
        bev_visual = (bev_map * 255).astype(np.uint8)
        
        # 转换为彩色图像
        bev_visual = cv2.cvtColor(bev_visual, cv2.COLOR_GRAY2BGR)
        
        return bev_visual
    
    def generate_multi_camera_bev(self, frame_images: Dict[str, np.ndarray], camera_detections: Dict[str, List[Dict]]) -> Tuple[np.ndarray, List[Dict]]:
        """
        生成多摄像头BEV
        
        Args:
            frame_images: 摄像头名称到图像的映射
            camera_detections: 摄像头名称到检测结果的映射
            
        Returns:
            融合后的BEV特征图和BEV中的目标信息
        """
        # 创建BEV特征图
        fused_bev = np.zeros(self.bev_size, dtype=np.float32)
        fused_objects = []
        
        # 处理每个摄像头
        for camera_name, image in frame_images.items():
            if camera_name not in self.homography_matrices:
                continue
            
            # 获取该摄像头的变换矩阵
            homography_matrix = self.homography_matrices[camera_name]
            
            # 处理该摄像头的检测结果
            if camera_name in camera_detections:
                detections = camera_detections[camera_name]
                for detection in detections:
                    # 获取检测框
                    x1, y1, x2, y2 = detection['bbox']
                    class_name = detection['class_name']
                    confidence = detection['confidence']
                    
                    # 计算检测框底部中心（假设目标在地面上）
                    bottom_center = np.array([(x1 + x2) / 2, y2], dtype=np.float32)
                    
                    # 转换到BEV坐标
                    bev_point = self._image_to_bev(bottom_center, homography_matrix)
                    
                    if bev_point is not None:
                        # 在BEV图上标记目标
                        self._mark_object(fused_bev, bev_point, class_name)
                        
                        # 保存BEV中的目标信息
                        bev_object = {
                            'class_name': class_name,
                            'bev_coordinates': bev_point.tolist(),
                            'confidence': confidence,
                            'camera': camera_name
                        }
                        fused_objects.append(bev_object)
        
        # 生成道路结构
        self._generate_road_structure(fused_bev)
        
        return fused_bev, fused_objects
    
    def _image_to_bev(self, point: np.ndarray, homography_matrix=None) -> Optional[np.ndarray]:
        """
        将图像坐标转换为BEV坐标
        
        Args:
            point: 图像坐标
            homography_matrix: 透视变换矩阵，默认为None，使用前视摄像头的变换矩阵
            
        Returns:
            BEV坐标，如果点在BEV范围内
        """
        # 使用提供的变换矩阵，或者使用默认值
        matrix = homography_matrix if homography_matrix is not None else self.homography_matrix
        
        # 添加齐次坐标
        point_hom = np.array([point[0], point[1], 1], dtype=np.float32)
        
        # 应用透视变换
        bev_point_hom = matrix @ point_hom
        bev_point_hom /= bev_point_hom[2]
        
        # 检查点是否在BEV范围内
        if (0 <= bev_point_hom[0] < self.bev_size[0] and 
            0 <= bev_point_hom[1] < self.bev_size[1]):
            return bev_point_hom[:2]
        else:
            return None

# 单元测试
if __name__ == "__main__":
    import os
    
    # 初始化BEV生成器
    bev_generator = BEVGenerator()
    
    # 测试图像路径
    test_image_path = "../../data/images/test.jpg"  # 请替换为实际测试图像路径
    
    if os.path.exists(test_image_path):
        print("Testing BEV generation...")
        
        # 加载图像
        image = cv2.imread(test_image_path)
        
        # 模拟检测结果
        mock_detections = [
            {
                'bbox': [200, 300, 400, 400],
                'class_name': 'car',
                'confidence': 0.9
            },
            {
                'bbox': [500, 350, 550, 420],
                'class_name': 'pedestrian',
                'confidence': 0.8
            }
        ]
        
        # 生成BEV
        bev_map, bev_objects = bev_generator.generate_bev(image, mock_detections)
        
        print(f"Generated BEV with {len(bev_objects)} objects")
        for obj in bev_objects:
            print(f"  - {obj['class_name']} at {obj['bev_coordinates']}")
        
        # 可视化BEV
        bev_visual = bev_generator.visualize_bev(bev_map)
        
        # 保存可视化结果
        output_path = "../../output/bev_test.jpg"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, bev_visual)
        print(f"BEV visualization saved to {output_path}")
    else:
        print(f"Test image not found: {test_image_path}")
        print("Please create a test image in the data/images directory.")
