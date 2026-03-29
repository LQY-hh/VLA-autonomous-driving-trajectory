import numpy as np
from typing import Dict, List, Optional, Any
import requests
import json
import time

class AdviceGenerator:
    """驾驶建议生成模块"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化驾驶建议生成器
        
        Args:
            config: 配置参数
        """
        # 默认配置
        default_config = {
            'speed_threshold': 10.0,  # 速度阈值（m/s）
            'distance_threshold': 5.0,  # 距离阈值（m）
            'angle_threshold': 0.3,  # 角度阈值（弧度）
            'driving_style': 'normal',  # 驾驶风格：normal, aggressive, conservative
            'pedestrian_distance_threshold': 8.0,  # 行人距离阈值
            'cyclist_distance_threshold': 6.0,  #  cyclists距离阈值
            'junction_distance_threshold': 10.0,  # 路口距离阈值
            'curve_radius_threshold': 50.0,  # 弯道半径阈值
            'weather_conditions': 'clear',  # 天气条件：clear, rain, fog, snow
            'use_llm': True,  # 是否使用大语言模型
            'llm_api_url': 'http://localhost:8000/generate',  # 大语言模型API地址
            'language': 'en',  # 语言：zh, en - 设置为英文以确保正确显示
            'road_type': 'urban',  # 道路类型：urban, highway, rural
            'scene_context': {}  # 场景上下文信息
        }
        
        # 更新配置
        self.config = default_config
        if config:
            self.config.update(config)
        
        self.speed_threshold = self.config['speed_threshold']
        self.distance_threshold = self.config['distance_threshold']
        self.angle_threshold = self.config['angle_threshold']
        self.driving_style = self.config['driving_style']
        self.pedestrian_distance_threshold = self.config['pedestrian_distance_threshold']
        self.cyclist_distance_threshold = self.config['cyclist_distance_threshold']
        self.junction_distance_threshold = self.config['junction_distance_threshold']
        self.curve_radius_threshold = self.config['curve_radius_threshold']
        self.weather_conditions = self.config['weather_conditions']
        self.use_llm = self.config['use_llm']
        self.llm_api_url = self.config['llm_api_url']
        self.language = self.config['language']
        self.road_type = self.config['road_type']
        self.scene_context = self.config['scene_context']
    
    def generate_advice(self, trajectory: List[List[float]], heading: List[float], 
                       bev_objects: List[Dict], current_speed: float = 0.0, 
                       scene_info: Optional[Dict] = None) -> Dict:
        """
        生成驾驶建议
        
        Args:
            trajectory: 预测轨迹
            heading: 航向信息
            bev_objects: BEV中的目标信息
            current_speed: 当前车速
            scene_info: 场景信息
            
        Returns:
            驾驶建议
        """
        # 分析轨迹
        trajectory_analysis = self._analyze_trajectory(trajectory, heading)
        
        # 分析场景
        scene_analysis = self._analyze_scene(bev_objects)
        
        # 生成建议
        advice = self._generate_advice(trajectory_analysis, scene_analysis, current_speed, scene_info)
        
        return advice
    
    def _analyze_trajectory(self, trajectory: List[List[float]], heading: List[float]) -> Dict:
        """
        分析轨迹
        
        Args:
            trajectory: 预测轨迹
            heading: 航向信息
            
        Returns:
            轨迹分析结果
        """
        # 计算轨迹长度
        trajectory_np = np.array(trajectory)
        displacement = np.linalg.norm(trajectory_np[-1] - trajectory_np[0])
        
        # 计算平均速度
        time_steps = len(trajectory) - 1
        if time_steps > 0:
            distances = np.linalg.norm(np.diff(trajectory_np, axis=0), axis=1)
            total_distance = np.sum(distances)
            average_speed = total_distance / time_steps  # 假设每个时间步为0.2秒
        else:
            average_speed = 0.0
        
        # 计算转向角度
        if len(heading) > 1:
            heading_changes = np.abs(np.diff(heading))
            max_heading_change = np.max(heading_changes)
            average_heading_change = np.mean(heading_changes)
        else:
            max_heading_change = 0.0
            average_heading_change = 0.0
        
        # 分析轨迹类型
        if max_heading_change < self.angle_threshold:
            trajectory_type = 'straight'
        else:
            if trajectory_np[-1, 0] > trajectory_np[0, 0]:
                trajectory_type = 'right'
            else:
                trajectory_type = 'left'
        
        # 计算弯道半径
        curve_radius = self._calculate_curve_radius(trajectory_np)
        
        # 分析是否接近路口
        near_junction = self._check_junction_proximity(trajectory_np)
        
        return {
            'displacement': float(displacement),
            'average_speed': float(average_speed),
            'max_heading_change': float(max_heading_change),
            'average_heading_change': float(average_heading_change),
            'trajectory_type': trajectory_type,
            'curve_radius': float(curve_radius),
            'near_junction': near_junction
        }
    
    def _calculate_curve_radius(self, trajectory: np.ndarray) -> float:
        """
        计算轨迹的弯道半径
        
        Args:
            trajectory: 轨迹坐标
            
        Returns:
            弯道半径
        """
        if len(trajectory) < 3:
            return float('inf')
        
        # 使用三点计算弯道半径
        p1, p2, p3 = trajectory[0], trajectory[len(trajectory)//2], trajectory[-1]
        
        # 计算圆心
        A = p2[0] - p1[0]
        B = p2[1] - p1[1]
        C = p3[0] - p1[0]
        D = p3[1] - p1[1]
        
        E = A * (p1[0] + p2[0]) + B * (p1[1] + p2[1])
        F = C * (p1[0] + p3[0]) + D * (p1[1] + p3[1])
        
        G = 2 * (A * (p3[1] - p2[1]) - B * (p3[0] - p2[0]))
        
        if G == 0:
            return float('inf')
        
        center_x = (D * E - B * F) / G
        center_y = (A * F - C * E) / G
        
        radius = np.sqrt((center_x - p1[0])**2 + (center_y - p1[1])**2)
        return radius
    
    def _check_junction_proximity(self, trajectory: np.ndarray) -> bool:
        """
        检查是否接近路口
        
        Args:
            trajectory: 轨迹坐标
            
        Returns:
            是否接近路口
        """
        # 这里简化处理，实际应用中可能需要更复杂的逻辑
        # 例如使用地图数据或检测路口特征
        return False
    
    def _analyze_scene(self, bev_objects: List[Dict]) -> Dict:
        """
        分析场景
        
        Args:
            bev_objects: BEV中的目标信息
            
        Returns:
            场景分析结果
        """
        # 统计目标数量
        object_counts = {}
        for obj in bev_objects:
            class_name = obj['class_name']
            if class_name not in object_counts:
                object_counts[class_name] = 0
            object_counts[class_name] += 1
        
        # 计算最近目标距离
        min_distance = float('inf')
        nearest_object = None
        for obj in bev_objects:
            x, y = obj['bev_coordinates']
            # 假设车辆在BEV图的中心底部
            distance = np.sqrt(x**2 + (y - 500)**2) * 0.1  # 转换为实际距离（米）
            if distance < min_distance:
                min_distance = distance
                nearest_object = obj
        
        # 分析场景类型
        if 'pedestrian' in object_counts:
            scene_type = 'pedestrian'
        elif 'cyclist' in object_counts:
            scene_type = 'cyclist'
        elif 'car' in object_counts:
            scene_type = 'traffic'
        elif 'truck' in object_counts:
            scene_type = 'traffic'
        else:
            scene_type = 'clear'
        
        # 分析目标风险等级
        risk_level = self._calculate_risk_level(object_counts, min_distance)
        
        # 分析交通密度
        traffic_density = self._calculate_traffic_density(object_counts)
        
        return {
            'object_counts': object_counts,
            'min_distance': float(min_distance) if min_distance != float('inf') else None,
            'nearest_object': nearest_object,
            'scene_type': scene_type,
            'risk_level': risk_level,
            'traffic_density': traffic_density
        }
    
    def _calculate_risk_level(self, object_counts: Dict, min_distance: float) -> str:
        """
        计算场景风险等级
        
        Args:
            object_counts: 目标数量
            min_distance: 最近目标距离
            
        Returns:
            风险等级
        """
        if min_distance is None:
            return 'low'
        
        # 基于距离和目标类型计算风险
        if min_distance < 2.0:
            return 'high'
        elif min_distance < 5.0:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_traffic_density(self, object_counts: Dict) -> str:
        """
        计算交通密度
        
        Args:
            object_counts: 目标数量
            
        Returns:
            交通密度等级
        """
        total_objects = sum(object_counts.values())
        if total_objects >= 5:
            return 'high'
        elif total_objects >= 2:
            return 'medium'
        else:
            return 'low'
    
    def _call_llm(self, prompt: str) -> str:
        """
        调用大语言模型API
        
        Args:
            prompt: 提示文本
            
        Returns:
            生成的文本
        """
        max_retries = 3
        for i in range(max_retries):
            try:
                response = requests.post(
                    self.llm_api_url,
                    json={'prompt': prompt},
                    timeout=5
                )
                if response.status_code == 200:
                    result = response.json()
                    generated_text = result.get('generated_text', '')
                    # 清理生成的文本
                    if generated_text:
                        # 去除首尾空白
                        generated_text = generated_text.strip()
                        # 去除多余的换行
                        generated_text = ' '.join(generated_text.split())
                        # 确保返回的是正确的中文编码
                        try:
                            generated_text = generated_text.encode('utf-8').decode('utf-8')
                        except:
                            generated_text = "保持安全驾驶，注意前方路况"
                    return generated_text
                else:
                    print(f"LLM API call failed with status code: {response.status_code}")
                    if i < max_retries - 1:
                        print(f"Retrying... ({i+1}/{max_retries})")
                        time.sleep(1)
                    else:
                        # 直接返回备选建议，不返回空字符串
                        return self._generate_fallback_advice(prompt)
            except Exception as e:
                print(f"Error calling LLM API: {e}")
                if i < max_retries - 1:
                    print(f"Retrying... ({i+1}/{max_retries})")
                    time.sleep(1)
                else:
                    # 本地备选方案：基于规则生成建议
                    return self._generate_fallback_advice(prompt)
    
    def _get_direction_text(self, trajectory_type: str) -> str:
        """
        获取方向文本描述
        
        Args:
            trajectory_type: 轨迹类型
            
        Returns:
            方向文本
        """
        direction_map = {
            'straight': '直行',
            'left': '向左转弯',
            'right': '向右转弯'
        }
        return direction_map.get(trajectory_type, '直行')
    
    def _get_scene_type_text(self, scene_type: str) -> str:
        """
        获取场景类型文本描述
        
        Args:
            scene_type: 场景类型
            
        Returns:
            场景类型文本
        """
        scene_map = {
            'pedestrian': '行人区域',
            'cyclist': '骑行者区域',
            'traffic': '交通繁忙区域',
            'clear': '空旷区域'
        }
        return scene_map.get(scene_type, '普通场景')
    
    def _get_risk_level_text(self, risk_level: str) -> str:
        """
        获取风险等级文本描述
        
        Args:
            risk_level: 风险等级
            
        Returns:
            风险等级文本
        """
        risk_map = {
            'high': '高风险',
            'medium': '中等风险',
            'low': '低风险'
        }
        return risk_map.get(risk_level, '未知风险')
    
    def _get_traffic_density_text(self, traffic_density: str) -> str:
        """
        获取交通密度文本描述
        
        Args:
            traffic_density: 交通密度
            
        Returns:
            交通密度文本
        """
        density_map = {
            'high': '高密度',
            'medium': '中等密度',
            'low': '低密度'
        }
        return density_map.get(traffic_density, '未知密度')
    
    def _get_object_name(self, object_name: str, english: bool = False) -> str:
        """
        获取物体名称的中文或英文描述
        
        Args:
            object_name: 物体名称
            english: 是否返回英文
            
        Returns:
            物体名称描述
        """
        if english:
            object_map = {
                'car': 'car',
                'truck': 'truck',
                'pedestrian': 'pedestrian',
                'cyclist': 'cyclist',
                'bus': 'bus',
                'motorcycle': 'motorcycle'
            }
        else:
            object_map = {
                'car': '汽车',
                'truck': '卡车',
                'pedestrian': '行人',
                'cyclist': '骑行者',
                'bus': '公交车',
                'motorcycle': '摩托车'
            }
        return object_map.get(object_name, object_name)
    
    def _generate_fallback_advice(self, prompt: str) -> str:
        """
        当LLM调用失败时的备选建议生成
        
        Args:
            prompt: 提示文本
            
        Returns:
            备选建议
        """
        # 基于更详细的规则生成备选建议
        if self.language == 'zh':
            if '行人' in prompt or 'pedestrian' in prompt:
                if '高风险' in prompt or 'high risk' in prompt:
                    return '注意行人，立即减速并准备停车'
                else:
                    return '注意行人，减速并保持安全距离'
            elif '弯道' in prompt or 'curve' in prompt:
                if '急弯' in prompt or 'sharp curve' in prompt:
                    return '前方急弯，显著减速并保持方向盘稳定'
                else:
                    return '前方弯道，减速并保持安全距离'
            elif '车辆' in prompt or 'car' in prompt:
                if '过近' in prompt or 'too close' in prompt:
                    return '前方车辆过近，紧急减速并保持安全距离'
                else:
                    return '保持安全距离，谨慎驾驶'
            elif '交通密集' in prompt or 'high density' in prompt:
                return '交通密集，减速并保持安全距离'
            elif '高速公路' in prompt or 'highway' in prompt:
                return '高速公路驾驶，保持恒定速度并注意前方路况'
            elif '城市道路' in prompt or 'urban' in prompt:
                return '城市道路驾驶，减速并注意行人'
            elif '乡村道路' in prompt or 'rural' in prompt:
                return '乡村道路驾驶，观察路况并保持安全速度'
            elif '雨天' in prompt or 'rain' in prompt:
                return '雨天路滑，减速并保持安全距离'
            elif '雾天' in prompt or 'fog' in prompt:
                return '雾天能见度低，开启雾灯并减速'
            elif '雪天' in prompt or 'snow' in prompt:
                return '雪天路滑，谨慎驾驶并保持安全距离'
            elif '直行' in prompt or 'straight' in prompt:
                return '保持直线行驶，注意前方路况'
            elif '左转' in prompt or 'left' in prompt:
                return '准备向左转弯，减速并观察路况'
            elif '右转' in prompt or 'right' in prompt:
                return '准备向右转弯，减速并观察路况'
            else:
                return '保持当前行驶状态，注意前方路况'
        else:
            if '行人' in prompt or 'pedestrian' in prompt:
                if '高风险' in prompt or 'high risk' in prompt:
                    return 'Watch out for pedestrians, immediately reduce speed and prepare to stop'
                else:
                    return 'Watch out for pedestrians, reduce speed and maintain a safe distance'
            elif '弯道' in prompt or 'curve' in prompt:
                if '急弯' in prompt or 'sharp curve' in prompt:
                    return 'Sharp curve ahead, significantly reduce speed and keep steering wheel stable'
                else:
                    return 'Curve ahead, reduce speed and maintain a safe distance'
            elif '车辆' in prompt or 'car' in prompt:
                if '过近' in prompt or 'too close' in prompt:
                    return 'Vehicle ahead too close, emergency reduce speed and maintain safe distance'
                else:
                    return 'Maintain safe distance, drive carefully'
            elif '交通密集' in prompt or 'high density' in prompt:
                return 'Traffic dense, reduce speed and maintain safe distance'
            elif '高速公路' in prompt or 'highway' in prompt:
                return 'Highway driving, maintain constant speed and pay attention to road conditions ahead'
            elif '城市道路' in prompt or 'urban' in prompt:
                return 'Urban road driving, reduce speed and watch out for pedestrians'
            elif '乡村道路' in prompt or 'rural' in prompt:
                return 'Rural road driving, observe road conditions and maintain safe speed'
            elif '雨天' in prompt or 'rain' in prompt:
                return 'Rainy road, reduce speed and maintain safe distance'
            elif '雾天' in prompt or 'fog' in prompt:
                return 'Foggy visibility low, turn on fog lights and reduce speed'
            elif '雪天' in prompt or 'snow' in prompt:
                return 'Snowy road, drive carefully and maintain safe distance'
            elif '直行' in prompt or 'straight' in prompt:
                return 'Keep driving straight, pay attention to road conditions ahead'
            elif '左转' in prompt or 'left' in prompt:
                return 'Prepare to turn left, reduce speed and observe road conditions'
            elif '右转' in prompt or 'right' in prompt:
                return 'Prepare to turn right, reduce speed and observe road conditions'
            else:
                return 'Maintain current driving state, pay attention to road conditions'
    
    def _generate_llm_prompt(self, trajectory_analysis: Dict, scene_analysis: Dict, 
                            current_speed: float) -> str:
        """
        生成大语言模型提示
        
        Args:
            trajectory_analysis: 轨迹分析结果
            scene_analysis: 场景分析结果
            current_speed: 当前车速
            
        Returns:
            提示文本
        """
        if self.language == 'zh':
            prompt = f"作为一名专业的自动驾驶系统驾驶顾问，请根据以下详细信息生成自然、准确、符合实际驾驶场景的中文驾驶建议：\n\n"
            
            # 道路类型和场景上下文
            prompt += f"【驾驶场景】\n"
            prompt += f"- 道路类型：{self.road_type}\n"
            prompt += f"- 天气条件：{self.weather_conditions}\n"
            prompt += f"- 驾驶风格：{self.driving_style}\n"
            if self.scene_context:
                prompt += f"- 附加信息：{self.scene_context}\n"
            prompt += "\n"
            
            # 轨迹分析
            prompt += f"【轨迹规划】\n"
            prompt += f"- 行驶方向：{self._get_direction_text(trajectory_analysis['trajectory_type'])}\n"
            prompt += f"- 预计平均速度：{trajectory_analysis['average_speed']:.1f} m/s ({trajectory_analysis['average_speed'] * 3.6:.1f} km/h)\n"
            if trajectory_analysis['curve_radius'] < self.curve_radius_threshold:
                prompt += f"- 前方弯道半径：{trajectory_analysis['curve_radius']:.1f} m\n"
            prompt += f"- 航向变化：{trajectory_analysis['max_heading_change']:.2f} rad\n"
            prompt += "\n"
            
            # 场景分析
            prompt += f"【周围环境】\n"
            prompt += f"- 场景类型：{self._get_scene_type_text(scene_analysis['scene_type'])}\n"
            prompt += f"- 风险等级：{self._get_risk_level_text(scene_analysis['risk_level'])}\n"
            prompt += f"- 交通密度：{self._get_traffic_density_text(scene_analysis['traffic_density'])}\n"
            
            if scene_analysis['min_distance']:
                prompt += f"- 最近障碍物距离：{scene_analysis['min_distance']:.1f} m\n"
            if scene_analysis['object_counts']:
                objects_text = ', '.join([f"{count}个{self._get_object_name(name)}" for name, count in scene_analysis['object_counts'].items()])
                prompt += f"- 周围障碍物：{objects_text}\n"
            prompt += "\n"
            
            # 当前状态
            prompt += f"【当前状态】\n"
            prompt += f"- 当前车速：{current_speed:.1f} m/s ({current_speed * 3.6:.1f} km/h)\n"
            prompt += "\n"
            
            # 生成要求
            prompt += "【生成要求】\n"
            prompt += "1. 语言：使用简洁、自然的中文表达\n"
            prompt += "2. 内容：基于上述信息，生成符合实际驾驶场景的建议\n"
            prompt += "3. 格式：直接给出建议，不包含任何解释性内容\n"
            prompt += "4. 重点：优先考虑安全，结合道路类型和驾驶风格\n"
            prompt += "5. 长度：控制在1-2句，不要过于冗长\n"
        else:
            prompt = f"As a professional driving advisor for autonomous driving systems, please generate natural, accurate, and contextually appropriate driving advice in English based on the following detailed information:\n\n"
            
            # Road type and scene context
            prompt += f"[Driving Scene]\n"
            prompt += f"- Road type: {self.road_type}\n"
            prompt += f"- Weather conditions: {self.weather_conditions}\n"
            prompt += f"- Driving style: {self.driving_style}\n"
            if self.scene_context:
                prompt += f"- Additional info: {self.scene_context}\n"
            prompt += "\n"
            
            # Trajectory analysis
            prompt += f"[Trajectory Planning]\n"
            prompt += f"- Direction: {self._get_direction_text(trajectory_analysis['trajectory_type'])}\n"
            prompt += f"- Expected average speed: {trajectory_analysis['average_speed']:.1f} m/s ({trajectory_analysis['average_speed'] * 3.6:.1f} km/h)\n"
            if trajectory_analysis['curve_radius'] < self.curve_radius_threshold:
                prompt += f"- Curve radius ahead: {trajectory_analysis['curve_radius']:.1f} m\n"
            prompt += f"- Heading change: {trajectory_analysis['max_heading_change']:.2f} rad\n"
            prompt += "\n"
            
            # Scene analysis
            prompt += f"[Surrounding Environment]\n"
            prompt += f"- Scene type: {self._get_scene_type_text(scene_analysis['scene_type'])}\n"
            prompt += f"- Risk level: {self._get_risk_level_text(scene_analysis['risk_level'])}\n"
            prompt += f"- Traffic density: {self._get_traffic_density_text(scene_analysis['traffic_density'])}\n"
            
            if scene_analysis['min_distance']:
                prompt += f"- Closest obstacle distance: {scene_analysis['min_distance']:.1f} m\n"
            if scene_analysis['object_counts']:
                objects_text = ', '.join([f"{count} {self._get_object_name(name, english=True)}" for name, count in scene_analysis['object_counts'].items()])
                prompt += f"- Surrounding obstacles: {objects_text}\n"
            prompt += "\n"
            
            # Current status
            prompt += f"[Current Status]\n"
            prompt += f"- Current speed: {current_speed:.1f} m/s ({current_speed * 3.6:.1f} km/h)\n"
            prompt += "\n"
            
            # Generation requirements
            prompt += "[Generation Requirements]\n"
            prompt += "1. Language: Use concise, natural English\n"
            prompt += "2. Content: Generate advice based on the above information\n"
            prompt += "3. Format: Direct advice without explanatory content\n"
            prompt += "4. Focus: Prioritize safety, considering road type and driving style\n"
            prompt += "5. Length: Keep it to 1-2 sentences, not too verbose\n"
        
        return prompt
    
    def _generate_advice(self, trajectory_analysis: Dict, scene_analysis: Dict, 
                        current_speed: float, scene_info: Optional[Dict] = None) -> Dict:
        """
        生成驾驶建议
        
        Args:
            trajectory_analysis: 轨迹分析结果
            scene_analysis: 场景分析结果
            current_speed: 当前车速
            scene_info: 场景信息
            
        Returns:
            驾驶建议
        """
        # 无论是否使用LLM，都先使用传统方法生成建议
        # 这样可以确保根据不同场景生成不同的建议
        return self._generate_traditional_advice(trajectory_analysis, scene_analysis, current_speed, scene_info)
    
    def _generate_traditional_advice(self, trajectory_analysis: Dict, scene_analysis: Dict, 
                                    current_speed: float, scene_info: Optional[Dict] = None) -> Dict:
        """
        使用传统方法生成驾驶建议
        
        Args:
            trajectory_analysis: 轨迹分析结果
            scene_analysis: 场景分析结果
            current_speed: 当前车速
            scene_info: 场景信息
            
        Returns:
            驾驶建议
        """
        advice_list = []
        confidence = 1.0
        
        # 获取场景信息
        road_type = self.road_type
        scene_context = self.scene_context.copy()
        
        if scene_info:
            road_type = scene_info.get('road_type', road_type)
            scene_context.update(scene_info.get('context', {}))
        
        # 基于道路类型的建议
        if road_type == 'highway':
            if current_speed < 20.0:  # 72 km/h
                advice_list.append('Maintain appropriate speed on highway')
            elif current_speed > 30.0:  # 108 km/h
                advice_list.append('Control speed on highway')
        elif road_type == 'urban':
            if current_speed > 15.0:  # 54 km/h
                advice_list.append('Reduce speed on urban road')
        elif road_type == 'rural':
            advice_list.append('Observe road conditions on rural road')
        
        # 基于轨迹类型的建议
        if trajectory_analysis['trajectory_type'] == 'straight':
            advice_list.append('Keep driving straight')
        elif trajectory_analysis['trajectory_type'] == 'left':
            advice_list.append('Turn left')
        elif trajectory_analysis['trajectory_type'] == 'right':
            advice_list.append('Turn right')
        
        # 基于弯道半径的建议
        if trajectory_analysis['curve_radius'] < self.curve_radius_threshold:
            curve_radius = trajectory_analysis['curve_radius']
            if curve_radius < self.curve_radius_threshold * 0.5:
                advice_list.append('Sharp curve ahead, reduce speed')
                confidence *= 0.8
            else:
                advice_list.append('Curve ahead, reduce speed')
                confidence *= 0.9
        
        # 基于速度的建议
        avg_speed = trajectory_analysis['average_speed']
        if avg_speed > self.speed_threshold * 1.2:
            advice_list.append('Speed too high, reduce speed')
            confidence *= 0.85
        elif avg_speed > self.speed_threshold:
            advice_list.append('Reduce speed')
            confidence *= 0.95
        elif avg_speed < self.speed_threshold * 0.3:
            advice_list.append('Speed too low, increase speed')
        elif avg_speed < self.speed_threshold * 0.5:
            advice_list.append('Increase speed')
        
        # 基于场景的建议
        if scene_analysis['scene_type'] == 'pedestrian':
            if scene_analysis['min_distance'] and scene_analysis['min_distance'] < self.pedestrian_distance_threshold:
                distance = scene_analysis['min_distance']
                if distance < 3.0:
                    advice_list.append('Watch out for pedestrians, immediately reduce speed')
                    confidence *= 0.7
                else:
                    advice_list.append('Watch out for pedestrians, reduce speed')
                    confidence *= 0.9
        elif scene_analysis['scene_type'] == 'cyclist':
            if scene_analysis['min_distance'] and scene_analysis['min_distance'] < self.cyclist_distance_threshold:
                advice_list.append('Watch out for cyclists, maintain safe distance')
                confidence *= 0.85
        elif scene_analysis['scene_type'] == 'traffic':
            if scene_analysis['min_distance'] and scene_analysis['min_distance'] < self.distance_threshold:
                distance = scene_analysis['min_distance']
                if distance < 2.0:
                    advice_list.append('Vehicle ahead too close, emergency reduce speed')
                    confidence *= 0.7
                else:
                    advice_list.append('Vehicle ahead, maintain safe distance')
                    confidence *= 0.8
        
        # 基于风险等级的建议
        if scene_analysis['risk_level'] == 'high':
            advice_list.append('High risk scene, drive carefully')
            confidence *= 0.7
        elif scene_analysis['risk_level'] == 'medium':
            advice_list.append('Medium risk scene, observe carefully')
            confidence *= 0.85
        
        # 基于交通密度的建议
        if scene_analysis['traffic_density'] == 'high':
            advice_list.append('Traffic dense, reduce speed')
        elif scene_analysis['traffic_density'] == 'medium':
            advice_list.append('Traffic moderate, maintain distance')
        
        # 基于天气条件的建议
        if self.weather_conditions == 'rain':
            advice_list.append('Rainy road, reduce speed')
            confidence *= 0.9
        elif self.weather_conditions == 'fog':
            advice_list.append('Foggy visibility low, turn on fog lights and reduce speed')
            confidence *= 0.8
        elif self.weather_conditions == 'snow':
            advice_list.append('Snowy road, drive carefully')
            confidence *= 0.75
        
        # 基于驾驶风格的调整
        if self.driving_style == 'aggressive':
            if 'Reduce speed' in advice_list:
                advice_list.remove('Reduce speed')
                advice_list.append('Moderately reduce speed')
            if 'Speed too high, reduce speed' in advice_list:
                advice_list.remove('Speed too high, reduce speed')
                advice_list.append('Moderately reduce speed')
        elif self.driving_style == 'conservative':
            if 'Increase speed' in advice_list:
                advice_list.remove('Increase speed')
                advice_list.append('Maintain current speed')
            if 'Speed too low, increase speed' in advice_list:
                advice_list.remove('Speed too low, increase speed')
                advice_list.append('Maintain current speed')
            advice_list.append('Maintain safe distance')
        
        # 基于场景上下文的建议
        if scene_context:
            if 'traffic_light' in scene_context:
                light_state = scene_context['traffic_light']
                if light_state == 'red':
                    advice_list.append('Red light, please stop')
                    confidence *= 0.95
                elif light_state == 'yellow':
                    advice_list.append('Yellow light, prepare to stop')
                    confidence *= 0.9
        
        # 生成最终建议
        if not advice_list:
            advice_list.append('Maintain current driving state')
        
        # 去重建议
        unique_advice = []
        seen = set()
        for advice in advice_list:
            if advice not in seen:
                seen.add(advice)
                unique_advice.append(advice)
        
        # 组合建议
        advice_text = '. '.join(unique_advice)
        
        return {
            'advice': advice_text,
            'confidence': confidence,
            'trajectory_analysis': trajectory_analysis,
            'scene_analysis': scene_analysis,
            'generation_method': 'traditional',
            'road_type': road_type,
            'timestamp': time.time()
        }

# 单元测试
if __name__ == "__main__":
    # 初始化建议生成器（传统方法）
    advice_generator = AdviceGenerator({'use_llm': False})
    
    # 测试数据
    trajectory = [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0], [7, 0], [8, 0], [9, 0]]
    heading = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    bev_objects = [
        {
            'class_name': 'car',
            'bev_coordinates': [250, 400],
            'confidence': 0.9
        }
    ]
    current_speed = 8.0
    
    # 测试场景信息
    scene_info = {
        'road_type': 'urban',
        'context': {'traffic_light': 'green'}
    }
    
    print("Testing traditional advice generation...")
    
    # 生成建议
    advice = advice_generator.generate_advice(trajectory, heading, bev_objects, current_speed, scene_info)
    
    print(f"Generated advice: {advice['advice']}")
    print(f"Confidence: {advice['confidence']:.2f}")
    print(f"Trajectory type: {advice['trajectory_analysis']['trajectory_type']}")
    print(f"Scene type: {advice['scene_analysis']['scene_type']}")
    print(f"Generation method: {advice.get('generation_method', 'traditional')}")
    print(f"Road type: {advice.get('road_type', 'unknown')}")
    print(f"Timestamp: {advice.get('timestamp', 'unknown')}")
    
    # 测试不同场景
    print("\nTesting with pedestrian scene...")
    bev_objects_pedestrian = [
        {
            'class_name': 'pedestrian',
            'bev_coordinates': [200, 450],
            'confidence': 0.8
        }
    ]
    advice_pedestrian = advice_generator.generate_advice(trajectory, heading, bev_objects_pedestrian, current_speed, scene_info)
    print(f"Generated advice: {advice_pedestrian['advice']}")
    print(f"Confidence: {advice_pedestrian['confidence']:.2f}")
    print(f"Scene type: {advice_pedestrian['scene_analysis']['scene_type']}")
    
    # 测试不同道路类型
    print("\nTesting with highway road type...")
    highway_scene_info = {
        'road_type': 'highway',
        'context': {}
    }
    advice_highway = advice_generator.generate_advice(trajectory, heading, bev_objects, 25.0, highway_scene_info)
    print(f"Generated advice: {advice_highway['advice']}")
    print(f"Road type: {advice_highway.get('road_type', 'unknown')}")
    
    # 测试LLM生成（禁用LLM，使用备选方案）
    print("\nTesting LLM advice generation...")
    llm_config = {
        'use_llm': False,  # 禁用LLM，使用备选方案
        'llm_api_url': 'http://localhost:8000/generate',
        'language': 'zh'
    }
    llm_advice_generator = AdviceGenerator(llm_config)
    
    try:
        llm_advice = llm_advice_generator.generate_advice(trajectory, heading, bev_objects, current_speed, scene_info)
        print(f"Generated advice: {llm_advice['advice']}")
        print(f"Confidence: {llm_advice['confidence']:.2f}")
        print(f"Generation method: {llm_advice.get('generation_method', 'unknown')}")
        print(f"Road type: {llm_advice.get('road_type', 'unknown')}")
    except Exception as e:
        print(f"LLM test failed: {e}")
        print("This is expected if no LLM server is running")
    
    # 测试不同语言
    print("\nTesting English advice generation...")
    en_config = {
        'language': 'en',
        'use_llm': False  # 禁用LLM，使用备选方案
    }
    en_advice_generator = AdviceGenerator(en_config)
    en_advice = en_advice_generator.generate_advice(trajectory, heading, bev_objects, current_speed, scene_info)
    print(f"Generated advice: {en_advice['advice']}")
    print(f"Confidence: {en_advice['confidence']:.2f}")
    
    # 测试弯道场景
    print("\nTesting curve scenario...")
    curve_trajectory = [[0, 0], [1, 0], [2, 1], [3, 2], [4, 4], [5, 6], [6, 8], [7, 10], [8, 12], [9, 14]]
    curve_heading = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    curve_advice = advice_generator.generate_advice(curve_trajectory, curve_heading, bev_objects, current_speed, scene_info)
    print(f"Generated advice: {curve_advice['advice']}")
    print(f"Trajectory type: {curve_advice['trajectory_analysis']['trajectory_type']}")
    print(f"Curve radius: {curve_advice['trajectory_analysis']['curve_radius']:.2f} m")
    
    print("\nAdvice generation testing completed!")
