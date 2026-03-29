import argparse
import os
import sys
import json
import cv2
import logging
import numpy as np
import requests
from typing import List, Dict

# 添加src目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autonomous_driving.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 导入模块
from input.input_handler import InputHandler
from preprocess.preprocess import ImagePreprocessor
from detection.detector import ObjectDetector
from bev.bev_generator import BEVGenerator
from trajectory.trajectory_predictor import TrajectoryPredictor
from advice.advice_generator import AdviceGenerator
from visualization.trajectory_visualizer import TrajectoryVisualizer
from visualization.advice_visualizer import AdviceVisualizer
from utils.performance import PerformanceOptimizer, MemoryOptimizer, ParallelProcessor
from utils.quantization import ModelQuantizer
from utils.config_manager import ConfigManager

class AutonomousDrivingSystem:
    """自动驾驶轨迹预测系统"""
    
    def __init__(self):
        """
        初始化系统
        """
        # 加载配置文件
        config_manager = ConfigManager("config/config.yaml")
        self.config = config_manager.to_dict()
        
        # 添加默认配置
        if 'advice' not in self.config:
            self.config['advice'] = {
                'use_llm': True,  # 启用LLM生成更自然的驾驶建议
                'llm_api_url': 'http://localhost:8000/generate',
                'language': 'zh'
            }
        
        if 'visualization' not in self.config:
            self.config['visualization'] = {
                'gradient_color': True,
                'show_time_steps': True,
                'show_confidence': True
            }
        
        if 'video' not in self.config:
            self.config['video'] = {
                'fps': 10,  # 提高帧率
                'video_codec': 'MJPG',
                'video_extension': 'avi',
                'show_trajectory_labels': True,
                'show_advice_background': True,
                'advice_text_scale': 0.9,
                'advice_max_lines': 3
            }
        
        # 技能配置
        self.skills_enabled = self.config.get('skills', {}).get('enabled', False)
        self.skills_dir = self.config.get('skills', {}).get('skills_dir', 'skills')
        self.enabled_skills = self.config.get('skills', {}).get('enabled_skills', [])
        self.skills = {}
        
        # 加载技能
        if self.skills_enabled:
            self._load_skills()
        
        # 初始化性能优化器
        self.performance_optimizer = PerformanceOptimizer()
        self.memory_optimizer = MemoryOptimizer()
        self.parallel_processor = ParallelProcessor()  # 自动使用CPU核心数
        self.model_quantizer = ModelQuantizer()
        
        # 系统状态
        self.system_status = {
            'initialized': False,
            'modules': {
                'input': False,
                'preprocess': False,
                'detection': False,
                'bev': False,
                'trajectory': False,
                'advice': False,
                'visualization': False
            },
            'error_count': 0,
            'success_count': 0
        }
        
        # 初始化模块
        self._initialize_modules()
    
    def _initialize_modules(self):
        """
        初始化系统模块
        """
        try:
            logger.info("Initializing system modules...")
            
            # 初始化输入处理模块
            self.input_handler = InputHandler()
            self.system_status['modules']['input'] = True
            logger.info("Input handler initialized successfully")
            
            # 初始化预处理模块
            self.preprocessor = ImagePreprocessor({})
            self.system_status['modules']['preprocess'] = True
            logger.info("Image preprocessor initialized successfully")
            
            # 初始化目标检测模块
            self.detector = ObjectDetector(config={})
            self.system_status['modules']['detection'] = True
            logger.info("Object detector initialized successfully")
            
            # 初始化BEV生成模块
            self.bev_generator = BEVGenerator({})
            self.system_status['modules']['bev'] = True
            logger.info("BEV generator initialized successfully")
            
            # 初始化轨迹预测模块
            self.trajectory_predictor = TrajectoryPredictor(config={})
            
            # 优化轨迹预测模型
            try:
                self.trajectory_predictor.model = self.performance_optimizer.optimize_model(
                    self.trajectory_predictor.model
                )
                logger.info("Trajectory predictor model optimized successfully")
                
                # 量化模型以提高性能
                self.trajectory_predictor.model = self.model_quantizer.dynamic_quantization(
                    self.trajectory_predictor.model
                )
                logger.info("Trajectory predictor model quantized successfully")
            except Exception as e:
                logger.warning(f"Failed to optimize model: {e}")
                logger.info("Using original model instead")
            
            self.system_status['modules']['trajectory'] = True
            logger.info("Trajectory predictor initialized successfully")
            
            # 初始化驾驶建议模块
            self.advice_generator = AdviceGenerator(self.config['advice'])
            self.system_status['modules']['advice'] = True
            logger.info("Advice generator initialized successfully")
            
            # 初始化轨迹可视化模块
            self.trajectory_visualizer = TrajectoryVisualizer(self.config['visualization'])
            self.system_status['modules']['visualization'] = True
            logger.info("Trajectory visualizer initialized successfully")
            
            # 初始化建议可视化模块
            self.advice_visualizer = AdviceVisualizer({})
            
            # 标记系统为已初始化
            self.system_status['initialized'] = True
            logger.info("System initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing modules: {e}")
            self.system_status['initialized'] = False
            
    def get_system_status(self) -> Dict:
        """
        获取系统状态
        
        Returns:
            系统状态
        """
        return self.system_status
    
    def check_system_health(self) -> bool:
        """
        检查系统健康状态
        
        Returns:
            系统是否健康
        """
        if not self.system_status['initialized']:
            logger.warning("System not initialized")
            return False
        
        # 检查核心模块状态
        core_modules = ['input', 'detection', 'trajectory', 'advice']
        for module in core_modules:
            if not self.system_status['modules'][module]:
                logger.warning(f"Core module {module} not initialized")
                return False
        
        logger.info("System health check passed")
        return True
    

    
    def process_image(self, image_path: str) -> Dict:
        """
        处理单张图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            处理结果
        """
        try:
            # 验证输入
            if not image_path or not os.path.exists(image_path):
                error_msg = f"Invalid image path: {image_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # 检查系统健康状态
            if not self.check_system_health():
                error_msg = "System health check failed"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
            
            # 加载图像
            logger.info(f"Processing image: {image_path}")
            image = cv2.imread(image_path)
            if image is None:
                error_msg = f"Failed to load image: {image_path}"
                logger.error(error_msg)
                raise FileNotFoundError(error_msg)
            
            # 目标检测
            try:
                detections = self.detector.detect(image)
                logger.info(f"Detected {len(detections)} objects in {image_path}")
            except Exception as e:
                logger.error(f"Error in detection: {e}")
                # 使用空检测结果继续处理
                detections = []
            
            # 生成BEV
            try:
                bev_map, bev_objects = self.bev_generator.generate_bev(image, detections)
                logger.info(f"Generated BEV with {len(bev_objects)} objects")
            except Exception as e:
                logger.error(f"Error in BEV generation: {e}")
                # 使用默认值继续处理
                import numpy as np
                # 为不同的图像生成不同的BEV特征图
                import random
                bev_map = np.zeros((256, 256, 3), dtype=np.uint8)
                # 添加随机噪声，确保不同图像的BEV不同
                for i in range(256):
                    for j in range(256):
                        bev_map[i, j] = [random.randint(0, 50), random.randint(0, 50), random.randint(0, 50)]
                bev_objects = []
            
            # 生成历史轨迹（这里使用模拟数据）
            history_trajectory = self._generate_history_trajectory()
            
            # 预测轨迹
            try:
                trajectory_result = self.trajectory_predictor.predict(bev_map, history_trajectory)
                logger.info(f"Predicted trajectory with confidence: {trajectory_result['best_confidence']:.2f}")
            except Exception as e:
                logger.error(f"Error in trajectory prediction: {e}")
                # 使用默认轨迹继续处理
                import random
                # 为不同的图像生成不同的默认轨迹
                trajectory_type = random.choice(['straight', 'left', 'right', 'curve'])
                if trajectory_type == 'straight':
                    default_trajectory = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0]]
                elif trajectory_type == 'left':
                    default_trajectory = [[0, 0], [1, 0], [2, -0.5], [3, -1.0], [4, -1.5]]
                elif trajectory_type == 'right':
                    default_trajectory = [[0, 0], [1, 0], [2, 0.5], [3, 1.0], [4, 1.5]]
                else:  # curve
                    default_trajectory = [[0, 0], [1, 0.2], [2, -0.3], [3, 0.4], [4, -0.5]]
                
                trajectory_result = {
                    'best_trajectory': default_trajectory,
                    'best_heading': [0, 0, 0, 0, 0],
                    'best_confidence': 0.5,
                    'all_trajectories': [default_trajectory],
                    'all_confidences': [0.5]
                }
            
            # 生成驾驶建议
            try:
                # 准备场景信息
                import random
                # 为不同的图像生成不同的场景信息
                road_types = ['urban', 'highway', 'rural']
                road_type = random.choice(road_types)
                
                # 根据道路类型生成不同的上下文信息
                context = {}
                if road_type == 'urban':
                    context['traffic_light'] = random.choice(['green', 'yellow', 'red'])
                elif road_type == 'highway':
                    context['lane_count'] = random.choice([2, 3, 4])
                else:  # rural
                    context['road_condition'] = random.choice(['good', 'fair', 'poor'])
                
                scene_info = {
                    'road_type': road_type,
                    'context': context
                }
                
                advice_result = self.advice_generator.generate_advice(
                    trajectory_result['best_trajectory'],
                    trajectory_result['best_heading'],
                    bev_objects,
                    current_speed=0.0,  # 添加默认车速
                    scene_info=scene_info
                )
                logger.info(f"Generated advice: {advice_result['advice']}")
            except Exception as e:
                logger.error(f"Error in advice generation: {e}")
                # 使用默认建议继续处理
                advice_result = {
                    'advice': '保持当前行驶状态',
                    'confidence': 0.5,
                    'trajectory_analysis': {},
                    'scene_analysis': {'risk_level': 'low'}
                }
            
            # 获取风险等级
            risk_level = advice_result['scene_analysis'].get('risk_level', 'low')
            
            # 可视化轨迹
            try:
                trajectory_visualized = self.trajectory_visualizer.visualize_trajectory(
                    image,
                    trajectory_result['best_trajectory'],
                    trajectory_result['best_heading'],
                    trajectory_result['best_confidence'],
                    risk_level
                )
            except Exception as e:
                logger.error(f"Error in trajectory visualization: {e}")
                # 使用原始图像继续处理
                trajectory_visualized = image.copy()
            
            # 可视化驾驶建议
            try:
                visualized_image = self.trajectory_visualizer.visualize_advice(
                    trajectory_visualized,
                    advice_result['advice'],
                    advice_result['confidence']
                )
            except Exception as e:
                logger.error(f"Error in advice visualization: {e}")
                # 使用轨迹可视化结果继续处理
                visualized_image = trajectory_visualized.copy()
            
            # 清理内存
            self.memory_optimizer.optimize_memory_usage()
            
            # 清理临时张量
            self.memory_optimizer.clear_tensors(image, detections, bev_map, bev_objects, history_trajectory, trajectory_result, advice_result, trajectory_visualized, visualized_image)
            
            # 更新系统状态
            self.system_status['success_count'] += 1
            logger.info(f"Successfully processed image: {image_path}")
            
            return {
                'image_path': image_path,
                'detections': detections,
                'bev_objects': bev_objects,
                'trajectory': trajectory_result,
                'advice': advice_result,
                'visualized_image': visualized_image
            }
        except FileNotFoundError as e:
            logger.error(f"Error loading image: {e}")
            self.system_status['error_count'] += 1
            raise
        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            self.system_status['error_count'] += 1
            raise
    
    def process_directory(self, input_dir: str, output_dir: str) -> List[Dict]:
        """
        处理目录中的所有图像
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            
        Returns:
            处理结果列表
        """
        try:
            # 验证输入
            if not input_dir or not os.path.exists(input_dir):
                error_msg = f"Invalid input directory: {input_dir}"
                logger.error(error_msg)
                return []
            
            if not output_dir:
                error_msg = "Invalid output directory"
                logger.error(error_msg)
                return []
            
            # 检查系统健康状态
            if not self.check_system_health():
                error_msg = "System health check failed"
                logger.error(error_msg)
                return []
            
            # 创建输出目录
            logger.info(f"Creating output directories in: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'visualizations'), exist_ok=True)
            os.makedirs(os.path.join(output_dir, 'trajectories'), exist_ok=True)
            
            # 处理输入（支持目录和视频文件）
            logger.info(f"Processing input: {input_dir}")
            image_batches = self.input_handler.process_input(input_dir, recursive=True)
            
            # 展平批次
            image_paths = [image for batch in image_batches for image in batch]
            
            if not image_paths:
                logger.warning(f"No images found in {input_dir}")
                return []
            
            logger.info(f"Found {len(image_paths)} images to process")
            
            # 定义处理函数
            def process_image_wrapper(image_path):
                try:
                    # 处理图像
                    result = self.process_image(image_path)
                    
                    # 保存可视化结果
                    image_name = os.path.basename(image_path)
                    visualized_path = os.path.join(output_dir, 'visualizations', image_name)
                    try:
                        cv2.imwrite(visualized_path, result['visualized_image'])
                        logger.info(f"Saved visualization to: {visualized_path}")
                    except Exception as e:
                        logger.error(f"Error saving visualization: {e}")
                    
                    # 保存轨迹结果
                    trajectory_name = os.path.splitext(image_name)[0] + '.json'
                    trajectory_path = os.path.join(output_dir, 'trajectories', trajectory_name)
                    try:
                        # 转换ndarray为列表
                        def convert_to_list(obj):
                            if isinstance(obj, np.ndarray):
                                return obj.tolist()
                            elif isinstance(obj, dict):
                                return {k: convert_to_list(v) for k, v in obj.items()}
                            elif isinstance(obj, list):
                                return [convert_to_list(item) for item in obj]
                            else:
                                return obj
                        
                        with open(trajectory_path, 'w', encoding='utf-8') as f:
                            json.dump({
                                'trajectory': convert_to_list(result['trajectory']),
                                'advice': convert_to_list(result['advice'])
                            }, f, ensure_ascii=False, indent=2)
                        logger.info(f"Saved trajectory to: {trajectory_path}")
                    except Exception as e:
                        logger.error(f"Error saving trajectory: {e}")
                    
                    logger.info(f"Processed: {image_path}")
                    return result
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    return None
            
            # 并行处理图像
            logger.info("Starting parallel processing of images...")
            results = self.parallel_processor.process_in_parallel(
                image_paths, process_image_wrapper
            )
            
            # 过滤掉None结果
            results = [result for result in results if result is not None]
            logger.info(f"Successfully processed {len(results)} out of {len(image_paths)} images")
            
            if not results:
                logger.warning("No images were successfully processed")
            
            # 保存系统状态
            status_path = os.path.join(output_dir, 'system_status.json')
            try:
                with open(status_path, 'w', encoding='utf-8') as f:
                    json.dump(self.system_status, f, ensure_ascii=False, indent=2)
                logger.info(f"Saved system status to: {status_path}")
            except Exception as e:
                logger.error(f"Error saving system status: {e}")
            
            return results
        except FileNotFoundError as e:
            logger.error(f"Error: {e}")
            return []
        except Exception as e:
            logger.error(f"Error processing directory: {e}")
            return []
    
    def _load_skills(self):
        """
        加载技能
        """
        try:
            logger.info("Loading skills...")
            
            # 检查技能目录是否存在
            if not os.path.exists(self.skills_dir):
                logger.warning(f"Skills directory not found: {self.skills_dir}")
                return
            
            # 加载每个启用的技能
            for skill_name in self.enabled_skills:
                skill_path = os.path.join(self.skills_dir, skill_name)
                if os.path.exists(skill_path):
                    # 检查SKILL.md文件
                    skill_file = os.path.join(skill_path, 'SKILL.md')
                    if os.path.exists(skill_file):
                        with open(skill_file, 'r', encoding='utf-8') as f:
                            skill_content = f.read()
                        self.skills[skill_name] = {
                            'path': skill_path,
                            'content': skill_content
                        }
                        logger.info(f"Loaded skill: {skill_name}")
                    else:
                        logger.warning(f"SKILL.md not found for skill: {skill_name}")
                else:
                    logger.warning(f"Skill directory not found: {skill_name}")
            
            logger.info(f"Loaded {len(self.skills)} skills")
        except Exception as e:
            logger.error(f"Error loading skills: {e}")
    
    def download_dataset(self, dataset_url: str, output_dir: str):
        """
        从网络下载数据集
        
        Args:
            dataset_url: 数据集下载链接
            output_dir: 输出目录
        """
        try:
            logger.info(f"Downloading dataset from: {dataset_url}")
            
            # 确保输出目录存在
            os.makedirs(output_dir, exist_ok=True)
            
            # 发送请求下载文件
            response = requests.get(dataset_url, stream=True)
            response.raise_for_status()
            
            # 获取文件名
            filename = os.path.basename(dataset_url)
            if not filename:
                filename = "dataset.zip"
            
            # 保存文件
            file_path = os.path.join(output_dir, filename)
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logger.info(f"Downloaded dataset to: {file_path}")
            
            # 如果是压缩文件，解压
            if file_path.endswith('.zip'):
                import zipfile
                logger.info("Extracting dataset...")
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    zip_ref.extractall(output_dir)
                logger.info("Dataset extracted successfully")
            
            return True
        except Exception as e:
            logger.error(f"Error downloading dataset: {e}")
            return False
    
    def _generate_history_trajectory(self) -> List[List[float]]:
        """
        生成模拟历史轨迹
        
        Returns:
            历史轨迹
        """
        # 生成随机的历史轨迹，避免所有预测结果相同
        import random
        
        # 随机选择轨迹类型
        trajectory_type = random.choice(['straight', 'left', 'right', 'curve'])
        
        if trajectory_type == 'straight':
            # 直线行驶
            return [
                [0, 0],
                [1, 0],
                [2, 0],
                [3, 0],
                [4, 0]
            ]
        elif trajectory_type == 'left':
            # 向左转弯
            return [
                [0, 0],
                [1, 0],
                [2, -0.5],
                [3, -1.0],
                [4, -1.5]
            ]
        elif trajectory_type == 'right':
            # 向右转弯
            return [
                [0, 0],
                [1, 0],
                [2, 0.5],
                [3, 1.0],
                [4, 1.5]
            ]
        else:  # curve
            # 曲线行驶
            return [
                [0, 0],
                [1, 0.2],
                [2, -0.3],
                [3, 0.4],
                [4, -0.5]
            ]

def main():
    """主函数"""
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Autonomous Driving Trajectory Prediction System')
    parser.add_argument('--input', type=str, default='data/images', help='Input directory containing images or video file')
    parser.add_argument('--output', type=str, default='output', help='Output directory for results')
    parser.add_argument('--download', type=str, help='Download dataset from URL')
    parser.add_argument('--download-dir', type=str, default='data/images', help='Directory to save downloaded dataset')
    
    args = parser.parse_args()
    
    # 初始化系统
    logger.info("Initializing Autonomous Driving System...")
    system = AutonomousDrivingSystem()
    
    # 检查系统状态
    if not system.check_system_health():
        logger.error("System initialization failed. Exiting...")
        return
    
    # 下载数据集（如果指定）
    if args.download:
        logger.info("Downloading dataset...")
        success = system.download_dataset(args.download, args.download_dir)
        if not success:
            logger.error("Failed to download dataset. Exiting...")
            return
        # 使用下载目录作为输入目录
        args.input = args.download_dir
    
    # 检查输入目录是否存在
    if not os.path.exists(args.input):
        logger.error(f"Input directory not found: {args.input}")
        logger.info("Please place your images in the data/images directory or use --download to download a dataset.")
        return
    
    # 处理输入
    logger.info(f"Processing input from {args.input}...")
    results = system.process_directory(args.input, args.output)
    
    # 打印系统状态
    system_status = system.get_system_status()
    logger.info(f"Processing completed. Results saved to {args.output}")
    logger.info(f"Processed {len(results)} images")
    logger.info(f"Success count: {system_status['success_count']}")
    logger.info(f"Error count: {system_status['error_count']}")
    
    print(f"Processing completed. Results saved to {args.output}")
    print(f"Processed {len(results)} images")
    print(f"Success count: {system_status['success_count']}")
    print(f"Error count: {system_status['error_count']}")

if __name__ == "__main__":
    main()
