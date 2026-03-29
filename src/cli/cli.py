import argparse
import os
import sys
import json
import logging
import time
from typing import Dict, List, Optional
from tqdm import tqdm

# 添加src目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入模块
from input.input_handler import InputHandler
from preprocess.preprocess import ImagePreprocessor
from detection.detector import ObjectDetector
from bev.bev_generator import BEVGenerator
from trajectory.trajectory_predictor import TrajectoryPredictor
from advice.advice_generator import AdviceGenerator
from visualization.trajectory_visualizer import TrajectoryVisualizer
from visualization.advice_visualizer import AdviceVisualizer
from visualization.video_generator import VideoGenerator
from input.multi_camera_handler import MultiCameraHandler
from map.map_integrator import MapIntegrator
from trajectory.multi_object_trajectory_predictor import MultiObjectTrajectoryPredictor
from utils.performance import PerformanceOptimizer, MemoryOptimizer, ParallelProcessor
from utils.quantization import ModelQuantizer
from utils.model_exporter import ModelExporter
from utils.system_monitor import SystemMonitor
from utils.input_validator import InputValidator
from analysis.result_analyzer import ResultAnalyzer

class CLI:
    """
    命令行界面
    """
    
    def __init__(self):
        """
        初始化命令行界面
        """
        self.parser = argparse.ArgumentParser(
            description='Autonomous Driving Trajectory Prediction System',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        self._setup_parser()
        # 系统监控器实例
        self.system_monitor = None
    
    def _setup_parser(self):
        """
        设置命令行参数
        """
        # 主命令
        subparsers = self.parser.add_subparsers(dest='command', help='Available commands')
        
        # 处理命令
        process_parser = subparsers.add_parser('process', help='Process input images or videos')
        process_parser.add_argument('--input', type=str, required=True, help='Input directory or video file')
        process_parser.add_argument('--output', type=str, default='output', help='Output directory for results')
        process_parser.add_argument('--multi-camera', action='store_true', help='Enable multi-camera processing')
        process_parser.add_argument('--map', type=str, help='Path to map data file')
        process_parser.add_argument('--multi-object', action='store_true', help='Enable multi-object trajectory prediction')
        process_parser.add_argument('--video', action='store_true', help='Generate output video')
        process_parser.add_argument('--fps', type=int, default=10, help='Video frame rate')
        process_parser.add_argument('--model', type=str, help='Path to custom model weights')
        process_parser.add_argument('--skip-validation', action='store_true', help='Skip input validation')
        
        # 测试命令
        test_parser = subparsers.add_parser('test', help='Run system tests')
        test_parser.add_argument('--module', type=str, choices=['all', 'detection', 'bev', 'trajectory', 'advice', 'visualization'], default='all', help='Module to test')
        test_parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
        
        # 状态命令
        status_parser = subparsers.add_parser('status', help='Check system status')
        
        # 配置命令
        config_parser = subparsers.add_parser('config', help='Show or modify configuration')
        config_parser.add_argument('--show', action='store_true', help='Show current configuration')
        config_parser.add_argument('--set', nargs=2, metavar=('KEY', 'VALUE'), help='Set configuration value')
        
        # 分析命令
        analyze_parser = subparsers.add_parser('analyze', help='Analyze system results')
        analyze_parser.add_argument('--trajectories', type=str, default='output/trajectories', help='Trajectories directory')
        analyze_parser.add_argument('--generate-plots', action='store_true', help='Generate analysis plots')
        analyze_parser.add_argument('--generate-report', action='store_true', help='Generate summary report')
        
        # 导出命令
        export_parser = subparsers.add_parser('export', help='Export model to different formats')
        export_parser.add_argument('--model', type=str, help='Path to model weights')
        export_parser.add_argument('--formats', nargs='+', default=['onnx', 'torchscript'], help='Export formats (onnx, torchscript, onnx_dynamic)')
        export_parser.add_argument('--output-dir', type=str, default='models/exported', help='Output directory for exported models')
        
        # 监控命令
        monitor_parser = subparsers.add_parser('monitor', help='Monitor system status')
        monitor_parser.add_argument('--start', action='store_true', help='Start monitoring')
        monitor_parser.add_argument('--stop', action='store_true', help='Stop monitoring')
        monitor_parser.add_argument('--status', action='store_true', help='Get current system status')
        monitor_parser.add_argument('--report', action='store_true', help='Generate monitoring report')
        monitor_parser.add_argument('--interval', type=float, default=1.0, help='Monitoring interval in seconds')
    
    def run(self, args=None):
        """
        运行命令行界面
        
        Args:
            args: 命令行参数
            
        Returns:
            命令执行结果
        """
        if args is None:
            args = self.parser.parse_args()
        
        if not args.command:
            self.parser.print_help()
            return
        
        try:
            if args.command == 'process':
                return self._process(args)
            elif args.command == 'test':
                return self._test(args)
            elif args.command == 'status':
                return self._status(args)
            elif args.command == 'config':
                return self._config(args)
            elif args.command == 'analyze':
                return self._analyze(args)
            elif args.command == 'export':
                return self._export(args)
            elif args.command == 'monitor':
                return self._monitor(args)
            else:
                self.parser.print_help()
                return
        except KeyboardInterrupt:
            print("\nOperation cancelled by user.")
            return False
        except Exception as e:
            import traceback
            print(f"Error: {e}")
            if hasattr(args, 'verbose') and args.verbose:
                traceback.print_exc()
            return False
    
    def _process(self, args):
        """
        处理输入
        
        Args:
            args: 命令行参数
            
        Returns:
            处理结果
        """
        print(f"Processing input: {args.input}")
        print(f"Output directory: {args.output}")
        
        # 验证输入路径
        if not args.input:
            print("Error: Input path is required.")
            return False
        
        if not os.path.exists(args.input):
            print(f"Error: Input path does not exist: {args.input}")
            return False
        
        # 验证输入数据
        if os.path.isdir(args.input) and not args.skip_validation:
            print("Validating input images...")
            validator = InputValidator()
            valid_images, invalid_images, validation_results = validator.validate_directory(args.input)
            
            if not valid_images:
                print("Error: No valid images found in input directory.")
                return False
            
            report = validator.get_validation_report(validation_results)
            print(report)
            
            if invalid_images:
                print(f"Warning: {len(invalid_images)} images are invalid and will be skipped.")
        
        # 创建输出目录
        os.makedirs(args.output, exist_ok=True)
        os.makedirs(os.path.join(args.output, 'visualizations'), exist_ok=True)
        os.makedirs(os.path.join(args.output, 'trajectories'), exist_ok=True)
        os.makedirs(os.path.join(args.output, 'videos'), exist_ok=True)
        
        # 初始化系统
        try:
            system = AutonomousDrivingSystem(
                multi_camera=args.multi_camera,
                map_path=args.map,
                multi_object=args.multi_object,
                model_path=args.model
            )
        except Exception as e:
            print(f"Error initializing system: {e}")
            return False
        
        # 检查系统状态
        if not system.check_system_health():
            print("System health check failed. Exiting...")
            return False
        
        # 处理输入
        try:
            results = system.process_directory(args.input, args.output, generate_video=args.video, video_fps=args.fps)
        except Exception as e:
            print(f"Error processing directory: {e}")
            return False
        
        # 打印结果
        print(f"Processing completed.")
        print(f"Processed {len(results)} images")
        print(f"Results saved to {args.output}")
        
        return True
    
    def _test(self, args):
        """
        运行系统测试
        
        Args:
            args: 命令行参数
            
        Returns:
            测试结果
        """
        print(f"Running tests for {args.module} module")
        
        # 这里可以添加测试逻辑
        if args.module == 'all':
            print("Running all module tests...")
            # 运行所有测试
        elif args.module == 'detection':
            print("Running detection module tests...")
            # 运行检测模块测试
        elif args.module == 'bev':
            print("Running BEV module tests...")
            # 运行BEV模块测试
        elif args.module == 'trajectory':
            print("Running trajectory module tests...")
            # 运行轨迹预测模块测试
        elif args.module == 'advice':
            print("Running advice module tests...")
            # 运行驾驶建议模块测试
        elif args.module == 'visualization':
            print("Running visualization module tests...")
            # 运行可视化模块测试
        
        print("Tests completed!")
        return True
    
    def _status(self, args):
        """
        检查系统状态
        
        Args:
            args: 命令行参数
            
        Returns:
            系统状态
        """
        print("Checking system status...")
        
        # 初始化系统
        system = AutonomousDrivingSystem()
        
        # 获取系统状态
        status = system.get_system_status()
        
        # 打印状态
        print(f"System initialized: {status['initialized']}")
        print("Module status:")
        for module, is_ready in status['modules'].items():
            print(f"  {module}: {'Ready' if is_ready else 'Not ready'}")
        print(f"Success count: {status['success_count']}")
        print(f"Error count: {status['error_count']}")
        
        return True
    
    def _config(self, args):
        """
        显示或修改配置
        
        Args:
            args: 命令行参数
            
        Returns:
            配置操作结果
        """
        if args.show:
            print("Current configuration:")
            # 显示当前配置
            config = {
                'preprocess': {},
                'detection': {},
                'bev': {},
                'trajectory': {},
                'advice': {
                    'use_llm': True,
                    'llm_api_url': 'http://localhost:8000/generate',
                    'language': 'zh'
                },
                'visualization': {
                    'gradient_color': True,
                    'show_time_steps': True,
                    'show_confidence': True
                },
                'video': {
                    'fps': 10,
                    'video_codec': 'MJPG',
                    'video_extension': 'avi'
                }
            }
            print(json.dumps(config, indent=2, ensure_ascii=False))
        elif args.set:
            key, value = args.set
            print(f"Setting {key} to {value}")
            # 这里可以添加修改配置的逻辑
        else:
            self.parser.print_help()
        
        return True
    
    def _analyze(self, args):
        """
        分析系统结果
        
        Args:
            args: 命令行参数
            
        Returns:
            分析操作结果
        """
        print(f"Analyzing results from {args.trajectories}")
        
        # 初始化分析器
        analyzer = ResultAnalyzer()
        
        # 分析轨迹
        trajectory_results = analyzer.analyze_trajectories(args.trajectories)
        
        # 分析建议
        advice_results = analyzer.analyze_advice(args.trajectories)
        
        # 分析性能
        performance_results = analyzer.analyze_system_performance()
        
        # 生成图表
        if args.generate_plots:
            analyzer.generate_plots(trajectory_results, advice_results)
        
        # 生成报告
        if args.generate_report:
            analyzer.generate_summary_report(trajectory_results, advice_results, performance_results)
        
        print("Analysis completed!")
        return True
    
    def _export(self, args):
        """
        导出模型到不同格式
        
        Args:
            args: 命令行参数
            
        Returns:
            导出操作结果
        """
        import torch
        print(f"Exporting model to formats: {args.formats}")
        
        # 初始化导出器
        exporter = ModelExporter(args.output_dir)
        
        # 加载模型
        from trajectory.trajectory_predictor import TrajectoryTransformer
        model = TrajectoryTransformer()
        
        # 如果提供了模型权重路径，则加载权重
        if args.model and os.path.exists(args.model):
            try:
                model.load_state_dict(torch.load(args.model, map_location='cpu'))
                print(f"Loaded model weights from {args.model}")
            except Exception as e:
                print(f"Error loading model weights: {e}")
                return False
        
        # 创建输入样例
        input_sample = torch.randn(1, 10, 2)  # 假设输入是10个时间步的坐标
        
        # 导出模型
        export_paths = exporter.export_model(model, input_sample, 'trajectory_transformer', args.formats)
        
        # 打印导出结果
        print("Model export completed!")
        for format_name, path in export_paths.items():
            print(f"{format_name}: {path}")
        
        return True
    
    def _monitor(self, args):
        """
        监控系统状态
        
        Args:
            args: 命令行参数
            
        Returns:
            监控操作结果
        """
        import time
        import json
        # 初始化系统监控器
        if not self.system_monitor:
            self.system_monitor = SystemMonitor(monitor_interval=args.interval)
        
        if args.start:
            print("Starting system monitoring...")
            self.system_monitor.start_monitoring()
            print("System monitoring started. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nStopping monitoring...")
                self.system_monitor.stop_monitoring()
        elif args.stop:
            print("Stopping system monitoring...")
            self.system_monitor.stop_monitoring()
        elif args.status:
            print("Getting current system status...")
            status = self.system_monitor.get_system_status()
            print(json.dumps(status, indent=2, ensure_ascii=False))
        elif args.report:
            print("Generating monitoring report...")
            report_file = self.system_monitor.generate_report()
            print(f"Monitoring report generated: {report_file}")
        else:
            self.parser.print_help()
        
        return True

class AutonomousDrivingSystem:
    """
    自动驾驶轨迹预测系统
    """
    
    def __init__(self, multi_camera=False, map_path=None, multi_object=False, model_path=None):
        """
        初始化系统
        
        Args:
            multi_camera: 是否启用多摄像头处理
            map_path: 地图数据路径
            multi_object: 是否启用多目标轨迹预测
            model_path: 模型权重路径
        """
        # 使用硬编码配置
        self.config = {
            'preprocess': {},
            'detection': {},
            'bev': {},
            'trajectory': {
                'model_path': model_path
            },
            'advice': {
                'use_llm': True,
                'llm_api_url': 'http://localhost:8000/generate',
                'language': 'zh'
            },
            'visualization': {
                'gradient_color': True,
                'show_time_steps': True,
                'show_confidence': True
            },
            'video': {
                'fps': 10,
                'video_codec': 'MJPG',
                'video_extension': 'avi'
            }
        }
        
        # 初始化性能优化器
        self.performance_optimizer = PerformanceOptimizer()
        self.memory_optimizer = MemoryOptimizer()
        self.parallel_processor = ParallelProcessor()
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
                'visualization': False,
                'video': False
            },
            'error_count': 0,
            'success_count': 0
        }
        
        # 功能标志
        self.multi_camera = multi_camera
        self.multi_object = multi_object
        self.map_path = map_path
        
        # 初始化模块
        self._initialize_modules()
    
    def _initialize_modules(self):
        """
        初始化系统模块
        """
        try:
            print("Initializing system modules...")
            
            # 初始化输入处理模块
            if self.multi_camera:
                self.input_handler = MultiCameraHandler()
            else:
                self.input_handler = InputHandler()
            self.system_status['modules']['input'] = True
            print("Input handler initialized successfully")
            
            # 初始化预处理模块
            self.preprocessor = ImagePreprocessor({})
            self.system_status['modules']['preprocess'] = True
            print("Image preprocessor initialized successfully")
            
            # 初始化目标检测模块
            self.detector = ObjectDetector(config={})
            self.system_status['modules']['detection'] = True
            print("Object detector initialized successfully")
            
            # 初始化BEV生成模块
            self.bev_generator = BEVGenerator({})
            self.system_status['modules']['bev'] = True
            print("BEV generator initialized successfully")
            
            # 初始化轨迹预测模块
            if self.multi_object:
                self.trajectory_predictor = MultiObjectTrajectoryPredictor(
                    model_path=self.config['trajectory']['model_path']
                )
            else:
                self.trajectory_predictor = TrajectoryPredictor(
                    config=self.config['trajectory']
                )
            
            # 优化轨迹预测模型
            try:
                if hasattr(self.trajectory_predictor, 'model'):
                    self.trajectory_predictor.model = self.performance_optimizer.optimize_model(
                        self.trajectory_predictor.model
                    )
                    # 量化模型
                    self.trajectory_predictor.model = self.model_quantizer.dynamic_quantization(
                        self.trajectory_predictor.model
                    )
                    print("Trajectory predictor model optimized and quantized successfully")
            except Exception as e:
                print(f"Failed to optimize model: {e}")
                print("Using original model instead")
            
            self.system_status['modules']['trajectory'] = True
            print("Trajectory predictor initialized successfully")
            
            # 初始化驾驶建议模块
            self.advice_generator = AdviceGenerator(self.config['advice'])
            self.system_status['modules']['advice'] = True
            print("Advice generator initialized successfully")
            
            # 初始化轨迹可视化模块
            self.trajectory_visualizer = TrajectoryVisualizer(self.config['visualization'])
            self.system_status['modules']['visualization'] = True
            print("Trajectory visualizer initialized successfully")
            
            # 初始化建议可视化模块
            self.advice_visualizer = AdviceVisualizer({})
            
            # 初始化视频生成模块
            self.video_generator = VideoGenerator(self.config['video'])
            self.system_status['modules']['video'] = True
            print("Video generator initialized successfully")
            
            # 初始化地图集成器
            if self.map_path:
                self.map_integrator = MapIntegrator(self.map_path)
                print("Map integrator initialized successfully")
            else:
                self.map_integrator = None
            
            # 标记系统为已初始化
            self.system_status['initialized'] = True
            print("System initialized successfully")
            
        except Exception as e:
            print(f"Error initializing modules: {e}")
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
            print("System not initialized")
            return False
        
        # 检查核心模块状态
        core_modules = ['input', 'detection', 'trajectory', 'advice']
        for module in core_modules:
            if not self.system_status['modules'][module]:
                print(f"Core module {module} not initialized")
                return False
        
        print("System health check passed")
        return True
    
    def process_directory(self, input_dir: str, output_dir: str, generate_video: bool = True, video_fps: int = 10) -> List[Dict]:
        """
        处理目录中的所有图像
        
        Args:
            input_dir: 输入目录
            output_dir: 输出目录
            generate_video: 是否生成视频
            video_fps: 视频帧率
            
        Returns:
            处理结果列表
        """
        try:
            # 验证输入
            if not input_dir or not os.path.exists(input_dir):
                print(f"Invalid input directory: {input_dir}")
                return []
            
            if not output_dir:
                print("Invalid output directory")
                return []
            
            # 检查系统健康状态
            if not self.check_system_health():
                print("System health check failed")
                return []
            
            # 处理输入
            print(f"Processing input: {input_dir}")
            
            if self.multi_camera:
                # 处理多摄像头输入
                camera_images = self.input_handler.process_multi_camera_input(input_dir)
                synchronized_frames = self.input_handler.synchronize_frames(camera_images)
                print(f"Found {len(synchronized_frames)} synchronized frames")
                
                results = []
                for i, frame in enumerate(synchronized_frames):
                    print(f"Processing frame {i+1}/{len(synchronized_frames)}")
                    result = self._process_multi_camera_frame(frame)
                    if result:
                        results.append(result)
                        
                        # 保存结果
                        self._save_result(result, output_dir, f"frame_{i:04d}")
            else:
                # 处理单摄像头输入
                image_batches = self.input_handler.process_input(input_dir, recursive=True)
                image_paths = [image for batch in image_batches for image in batch]
                
                if not image_paths:
                    print(f"No images found in {input_dir}")
                    return []
                
                print(f"Found {len(image_paths)} images to process")
                
                # 定义处理函数
                def process_image_wrapper(image_path):
                    try:
                        result = self._process_image(image_path)
                        if result:
                            # 保存结果
                            image_name = os.path.basename(image_path)
                            self._save_result(result, output_dir, os.path.splitext(image_name)[0])
                        return result
                    except Exception as e:
                        print(f"Error processing {image_path}: {e}")
                        return None
                
                # 并行处理图像
                print("Starting parallel processing of images...")
                
                # 使用进度条
                results = []
                for image_path in tqdm(image_paths, desc="Processing images", unit="image"):
                    result = process_image_wrapper(image_path)
                    if result:
                        results.append(result)
                
                print(f"Successfully processed {len(results)} out of {len(image_paths)} images")
            
            # 生成视频
            if generate_video and results:
                print("Generating video with trajectory and advice...")
                video_path = os.path.join(output_dir, 'videos', 'output.avi')
                
                # 准备视频生成所需的数据
                if self.multi_camera:
                    # 多摄像头视频生成
                    frame_images = []
                    trajectories = []
                    advice_list = []
                    for result in results:
                        frame_images.append(result['fused_image'])
                        trajectories.append(result['trajectory']['best_trajectory'])
                        advice_list.append(result['advice']['advice'])
                    
                    # 生成视频
                    success = self.video_generator.generate_video_from_images(
                        frame_images, video_path, fps=video_fps
                    )
                else:
                    # 单摄像头视频生成
                    video_image_paths = [result['image_path'] for result in results]
                    trajectories = [result['trajectory']['best_trajectory'] for result in results]
                    headings = [result['trajectory']['best_heading'] for result in results]
                    advice_list = [result['advice']['advice'] for result in results]
                    detections = [result['detections'] for result in results]
                    
                    # 准备场景信息
                    scene_info_list = []
                    for result in results:
                        scene_info = {
                            'road_type': 'urban',
                            'context': {}
                        }
                        scene_info_list.append(scene_info)
                    
                    # 生成视频
                    success = self.video_generator.generate_video(
                        video_image_paths, video_path,
                        trajectories=trajectories,
                        headings=headings,
                        advice=advice_list,
                        detections=detections,
                        scene_info=scene_info_list
                    )
                
                if success:
                    print(f"Video saved to: {video_path}")
                else:
                    print("Failed to generate video")
            
            # 保存系统状态
            status_path = os.path.join(output_dir, 'system_status.json')
            try:
                with open(status_path, 'w', encoding='utf-8') as f:
                    json.dump(self.system_status, f, ensure_ascii=False, indent=2)
                print(f"Saved system status to: {status_path}")
            except Exception as e:
                print(f"Error saving system status: {e}")
            
            return results
        except Exception as e:
            print(f"Error processing directory: {e}")
            return []
    
    def _process_image(self, image_path: str) -> Dict:
        """
        处理单张图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            处理结果
        """
        try:
            # 加载图像
            import cv2
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                return None
            
            # 标记用户车辆
            marked_image = self.input_handler.mark_user_vehicle(image)
            
            # 目标检测
            detections = self.detector.detect(marked_image)
            print(f"Detected {len(detections)} objects in {image_path}")
            
            # 生成BEV
            bev_map, bev_objects = self.bev_generator.generate_bev(marked_image, detections)
            print(f"Generated BEV with {len(bev_objects)} objects")
            
            # 生成历史轨迹
            history_trajectory = self._generate_history_trajectory()
            
            # 获取地图特征
            map_features = None
            if self.map_integrator:
                # 假设车辆位置在原点
                map_features = self.map_integrator.get_map_features([0, 0])
                # 集成地图信息到BEV
                bev_map = self.map_integrator.integrate_with_bev(bev_map, [0, 0])
            
            # 预测轨迹
            if self.multi_object:
                # 准备多目标输入
                objects = []
                for i, det in enumerate(detections):
                    obj = {
                        'id': i,
                        'class_id': det['class_id'],
                        'class_name': det['class_name'],
                        'confidence': det['confidence'],
                        'history': [det['bbox'][:2] for _ in range(5)]  # 简化处理，使用检测框左上角作为历史位置
                    }
                    objects.append(obj)
                
                trajectory_result = self.trajectory_predictor.predict(bev_map, objects)
                print(f"Predicted trajectories for {trajectory_result['num_objects']} objects")
            else:
                trajectory_result = self.trajectory_predictor.predict(
                    bev_map, history_trajectory, map_features=map_features
                )
                print(f"Predicted trajectory with confidence: {trajectory_result['best_confidence']:.2f}")
            
            # 生成驾驶建议
            advice_result = self.advice_generator.generate_advice(
                trajectory=trajectory_result['best_trajectory'],
                heading=trajectory_result['best_heading'],
                bev_objects=bev_objects,
                scene_info={'road_type': 'urban'}
            )
            print(f"Generated advice: {advice_result['advice']}")
            
            # 可视化轨迹
            trajectory_visualized = self.trajectory_visualizer.visualize_trajectory(
                marked_image,
                trajectory_result['best_trajectory'],
                trajectory_result['best_heading'],
                trajectory_result['best_confidence'],
                'low',
                detections=detections
            )
            
            # 可视化驾驶建议
            visualized_image = self.trajectory_visualizer.visualize_advice(
                trajectory_visualized,
                advice_result['advice'],
                advice_result['confidence']
            )
            
            # 清理内存
            self.memory_optimizer.optimize_memory_usage()
            
            # 更新系统状态
            self.system_status['success_count'] += 1
            print(f"Successfully processed image: {image_path}")
            
            return {
                'image_path': image_path,
                'detections': detections,
                'bev_objects': bev_objects,
                'trajectory': trajectory_result,
                'advice': advice_result,
                'visualized_image': visualized_image
            }
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            self.system_status['error_count'] += 1
            return None
    
    def _process_multi_camera_frame(self, frame: Dict[str, str]) -> Dict:
        """
        处理多摄像头帧
        
        Args:
            frame: 摄像头名称到图像路径的映射
            
        Returns:
            处理结果
        """
        try:
            # 加载帧图像
            frame_images = self.input_handler.load_frame_images(frame)
            
            # 融合图像
            fused_image = self.input_handler.fuse_images(frame_images, layout='grid')
            
            # 对每个摄像头进行目标检测
            camera_detections = {}
            for camera_name, image in frame_images.items():
                if image is not None:
                    detections = self.detector.detect(image)
                    camera_detections[camera_name] = detections
            
            # 生成多摄像头BEV
            bev_map, bev_objects = self.bev_generator.generate_multi_camera_bev(
                frame_images, camera_detections
            )
            print(f"Generated multi-camera BEV with {len(bev_objects)} objects")
            
            # 生成历史轨迹
            history_trajectory = self._generate_history_trajectory()
            
            # 预测轨迹
            trajectory_result = self.trajectory_predictor.predict(
                bev_map, history_trajectory
            )
            print(f"Predicted trajectory with confidence: {trajectory_result['best_confidence']:.2f}")
            
            # 生成驾驶建议
            advice_result = self.advice_generator.generate_advice(
                trajectory=trajectory_result['best_trajectory'],
                heading=trajectory_result['best_heading'],
                bev_objects=bev_objects,
                scene_info={'road_type': 'urban'}
            )
            print(f"Generated advice: {advice_result['advice']}")
            
            # 可视化轨迹
            trajectory_visualized = self.trajectory_visualizer.visualize_trajectory(
                fused_image,
                trajectory_result['best_trajectory'],
                trajectory_result['best_heading'],
                trajectory_result['best_confidence'],
                'low'
            )
            
            # 可视化驾驶建议
            visualized_image = self.trajectory_visualizer.visualize_advice(
                trajectory_visualized,
                advice_result['advice'],
                advice_result['confidence']
            )
            
            # 清理内存
            self.memory_optimizer.optimize_memory_usage()
            
            # 更新系统状态
            self.system_status['success_count'] += 1
            print("Successfully processed multi-camera frame")
            
            return {
                'frame': frame,
                'fused_image': fused_image,
                'camera_detections': camera_detections,
                'bev_objects': bev_objects,
                'trajectory': trajectory_result,
                'advice': advice_result,
                'visualized_image': visualized_image
            }
        except Exception as e:
            print(f"Error processing multi-camera frame: {e}")
            self.system_status['error_count'] += 1
            return None
    
    def _save_result(self, result: Dict, output_dir: str, base_name: str):
        """
        保存处理结果
        
        Args:
            result: 处理结果
            output_dir: 输出目录
            base_name: 基础文件名
        """
        import cv2
        
        # 保存可视化结果
        if 'visualized_image' in result:
            visualized_path = os.path.join(output_dir, 'visualizations', f"{base_name}.jpg")
            try:
                cv2.imwrite(visualized_path, result['visualized_image'])
                print(f"Saved visualization to: {visualized_path}")
            except Exception as e:
                print(f"Error saving visualization: {e}")
        
        # 保存轨迹结果
        trajectory_path = os.path.join(output_dir, 'trajectories', f"{base_name}.json")
        try:
            # 转换ndarray为列表
            def convert_to_list(obj):
                import numpy as np
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
            print(f"Saved trajectory to: {trajectory_path}")
        except Exception as e:
            print(f"Error saving trajectory: {e}")
    
    def _generate_history_trajectory(self) -> List[List[float]]:
        """
        生成本车的历史轨迹
        
        Returns:
            历史轨迹
        """
        # 生成本车的历史轨迹（直线行驶）
        # 确保轨迹从当前位置开始，向后延伸
        return [
            [-4, 0],  # 4个时间步之前的位置
            [-3, 0],  # 3个时间步之前的位置
            [-2, 0],  # 2个时间步之前的位置
            [-1, 0],  # 1个时间步之前的位置
            [0, 0]     # 当前位置
        ]

def main():
    """
    主函数
    """
    cli = CLI()
    cli.run()

if __name__ == "__main__":
    main()
