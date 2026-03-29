import os
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from input.input_handler import InputHandler

class MultiCameraHandler:
    """
    多摄像头输入处理器
    """
    
    def __init__(self):
        """
        初始化多摄像头处理器
        """
        self.input_handler = InputHandler()
        self.camera_configs = {
            'front': {'name': '前视摄像头', 'position': 'front', 'fov': 90},
            'rear': {'name': '后视摄像头', 'position': 'rear', 'fov': 120},
            'left': {'name': '左视摄像头', 'position': 'left', 'fov': 120},
            'right': {'name': '右视摄像头', 'position': 'right', 'fov': 120}
        }
    
    def process_multi_camera_input(self, input_dir: str) -> Dict[str, List[str]]:
        """
        处理多摄像头输入
        
        Args:
            input_dir: 输入目录，包含多个摄像头的子目录
            
        Returns:
            摄像头名称到图像路径列表的映射
        """
        camera_images = {}
        
        # 遍历输入目录
        for camera_name in self.camera_configs.keys():
            camera_dir = os.path.join(input_dir, camera_name)
            if os.path.exists(camera_dir):
                # 处理摄像头目录
                image_batches = self.input_handler.process_input(camera_dir, recursive=True)
                # 展平批次
                images = [image for batch in image_batches for image in batch]
                camera_images[camera_name] = images
        
        return camera_images
    
    def synchronize_frames(self, camera_images: Dict[str, List[str]]) -> List[Dict[str, str]]:
        """
        同步不同摄像头的帧
        
        Args:
            camera_images: 摄像头名称到图像路径列表的映射
            
        Returns:
            同步后的帧列表，每个元素是摄像头名称到图像路径的映射
        """
        if not camera_images:
            return []
        
        # 获取所有摄像头的图像数量
        min_frames = min(len(images) for images in camera_images.values())
        
        synchronized_frames = []
        for i in range(min_frames):
            frame = {}
            for camera_name, images in camera_images.items():
                if i < len(images):
                    frame[camera_name] = images[i]
            synchronized_frames.append(frame)
        
        return synchronized_frames
    
    def load_frame_images(self, frame: Dict[str, str]) -> Dict[str, Optional[np.ndarray]]:
        """
        加载帧中的所有图像
        
        Args:
            frame: 摄像头名称到图像路径的映射
            
        Returns:
            摄像头名称到图像的映射
        """
        frame_images = {}
        
        for camera_name, image_path in frame.items():
            try:
                image = cv2.imread(image_path)
                if image is not None:
                    frame_images[camera_name] = image
                else:
                    frame_images[camera_name] = None
            except Exception as e:
                print(f"Error loading image {image_path}: {e}")
                frame_images[camera_name] = None
        
        return frame_images
    
    def fuse_images(self, frame_images: Dict[str, np.ndarray], layout: str = 'grid') -> np.ndarray:
        """
        融合多个摄像头的图像
        
        Args:
            frame_images: 摄像头名称到图像的映射
            layout: 布局方式，支持 'grid'（网格）、'panorama'（全景）
            
        Returns:
            融合后的图像
        """
        if not frame_images:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        if layout == 'grid':
            return self._create_grid_layout(frame_images)
        elif layout == 'panorama':
            return self._create_panorama(frame_images)
        else:
            return self._create_grid_layout(frame_images)
    
    def _create_grid_layout(self, frame_images: Dict[str, np.ndarray]) -> np.ndarray:
        """
        创建网格布局的融合图像
        
        Args:
            frame_images: 摄像头名称到图像的映射
            
        Returns:
            融合后的图像
        """
        # 确定网格大小
        num_cameras = len(frame_images)
        if num_cameras == 1:
            # 单个摄像头，直接返回
            return list(frame_images.values())[0]
        elif num_cameras == 2:
            # 两个摄像头，水平排列
            images = list(frame_images.values())
            # 调整图像大小
            h1, w1 = images[0].shape[:2]
            h2, w2 = images[1].shape[:2]
            min_h = min(h1, h2)
            min_w = min(w1, w2)
            
            resized_images = []
            for img in images:
                resized = cv2.resize(img, (min_w, min_h))
                resized_images.append(resized)
            
            # 水平拼接
            return np.hstack(resized_images)
        elif num_cameras == 3:
            # 三个摄像头，2x2网格，右下角留空
            images = list(frame_images.values())
            # 调整图像大小
            min_h = min(img.shape[0] for img in images)
            min_w = min(img.shape[1] for img in images)
            
            resized_images = []
            for img in images:
                resized = cv2.resize(img, (min_w, min_h))
                resized_images.append(resized)
            
            # 创建2x2网格
            empty_img = np.zeros((min_h, min_w, 3), dtype=np.uint8)
            row1 = np.hstack([resized_images[0], resized_images[1]])
            row2 = np.hstack([resized_images[2], empty_img])
            return np.vstack([row1, row2])
        else:
            # 四个摄像头，2x2网格
            images = list(frame_images.values())
            # 调整图像大小
            min_h = min(img.shape[0] for img in images)
            min_w = min(img.shape[1] for img in images)
            
            resized_images = []
            for img in images:
                resized = cv2.resize(img, (min_w, min_h))
                resized_images.append(resized)
            
            # 创建2x2网格
            row1 = np.hstack([resized_images[0], resized_images[1]])
            row2 = np.hstack([resized_images[2], resized_images[3]])
            return np.vstack([row1, row2])
    
    def _create_panorama(self, frame_images: Dict[str, np.ndarray]) -> np.ndarray:
        """
        创建全景布局的融合图像
        
        Args:
            frame_images: 摄像头名称到图像的映射
            
        Returns:
            融合后的图像
        """
        # 简单实现：水平拼接前、左、右摄像头的图像
        front_img = frame_images.get('front')
        left_img = frame_images.get('left')
        right_img = frame_images.get('right')
        
        if front_img is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        
        # 调整图像大小
        h, w = front_img.shape[:2]
        
        # 调整左右摄像头图像大小
        if left_img is not None:
            left_img = cv2.resize(left_img, (w//2, h))
        else:
            left_img = np.zeros((h, w//2, 3), dtype=np.uint8)
        
        if right_img is not None:
            right_img = cv2.resize(right_img, (w//2, h))
        else:
            right_img = np.zeros((h, w//2, 3), dtype=np.uint8)
        
        # 水平拼接
        panorama = np.hstack([left_img, front_img, right_img])
        return panorama
    
    def get_camera_intrinsics(self, camera_name: str) -> Dict[str, np.ndarray]:
        """
        获取摄像头内参
        
        Args:
            camera_name: 摄像头名称
            
        Returns:
            摄像头内参
        """
        # 这里使用默认内参，实际应用中应该使用标定后的内参
        intrinsics = {
            'camera_matrix': np.array([[800, 0, 320], [0, 800, 240], [0, 0, 1]]),
            'dist_coeffs': np.zeros((5, 1))
        }
        return intrinsics
    
    def get_camera_extrinsics(self, camera_name: str) -> Dict[str, np.ndarray]:
        """
        获取摄像头外参
        
        Args:
            camera_name: 摄像头名称
            
        Returns:
            摄像头外参
        """
        # 这里使用默认外参，实际应用中应该使用标定后的外参
        extrinsics = {
            'rotation': np.eye(3),
            'translation': np.zeros((3, 1))
        }
        
        if camera_name == 'front':
            extrinsics['translation'] = np.array([[0], [0], [1]])
        elif camera_name == 'rear':
            extrinsics['rotation'] = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]])
            extrinsics['translation'] = np.array([[0], [0], [-1]])
        elif camera_name == 'left':
            extrinsics['rotation'] = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
            extrinsics['translation'] = np.array([[-1], [0], [0]])
        elif camera_name == 'right':
            extrinsics['rotation'] = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
            extrinsics['translation'] = np.array([[1], [0], [0]])
        
        return extrinsics
    
    def undistort_image(self, image: np.ndarray, camera_name: str) -> np.ndarray:
        """
        对图像进行去畸变
        
        Args:
            image: 输入图像
            camera_name: 摄像头名称
            
        Returns:
            去畸变后的图像
        """
        intrinsics = self.get_camera_intrinsics(camera_name)
        camera_matrix = intrinsics['camera_matrix']
        dist_coeffs = intrinsics['dist_coeffs']
        
        # 去畸变
        undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)
        return undistorted

if __name__ == "__main__":
    # 测试多摄像头处理器
    multi_camera_handler = MultiCameraHandler()
    
    # 测试处理多摄像头输入
    test_dir = "test_multi_camera"
    if os.path.exists(test_dir):
        camera_images = multi_camera_handler.process_multi_camera_input(test_dir)
        print(f"Found images for cameras: {list(camera_images.keys())}")
        
        # 测试同步帧
        synchronized_frames = multi_camera_handler.synchronize_frames(camera_images)
        print(f"Synchronized {len(synchronized_frames)} frames")
        
        # 测试加载和融合图像
        if synchronized_frames:
            frame = synchronized_frames[0]
            frame_images = multi_camera_handler.load_frame_images(frame)
            
            # 测试网格布局
            grid_fused = multi_camera_handler.fuse_images(frame_images, layout='grid')
            cv2.imwrite("fused_grid.jpg", grid_fused)
            print("Saved grid fused image to fused_grid.jpg")
            
            # 测试全景布局
            panorama_fused = multi_camera_handler.fuse_images(frame_images, layout='panorama')
            cv2.imwrite("fused_panorama.jpg", panorama_fused)
            print("Saved panorama fused image to fused_panorama.jpg")
    else:
        print(f"Test directory {test_dir} does not exist")
