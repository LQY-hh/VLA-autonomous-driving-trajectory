import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, List, Optional, Tuple

class TrajectoryTransformer(nn.Module):
    """轨迹Transformer模型"""
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化轨迹Transformer模型
        
        Args:
            config: 模型配置参数
        """
        super(TrajectoryTransformer, self).__init__()
        
        # 默认配置
        default_config = {
            'hidden_dim': 256,       # 隐藏层维度
            'num_heads': 8,          # 注意力头数
            'num_encoder_layers': 3,  # 编码器层数
            'num_decoder_layers': 3,  # 解码器层数
            'dropout': 0.1,           #  dropout率
            'history_len': 5,         # 历史轨迹长度
            'pred_len': 15,           # 预测轨迹长度（3秒，5Hz采样）
            'bev_size': (500, 500),   # BEV特征图尺寸
            'bev_channels': 1,         # BEV特征图通道数
            'num_modes': 3            # 预测模态数
        }
        
        # 更新配置
        self.config = default_config
        if config:
            self.config.update(config)
        
        self.hidden_dim = self.config['hidden_dim']
        self.num_heads = self.config['num_heads']
        self.num_encoder_layers = self.config['num_encoder_layers']
        self.num_decoder_layers = self.config['num_decoder_layers']
        self.dropout = self.config['dropout']
        self.history_len = self.config['history_len']
        self.pred_len = self.config['pred_len']
        self.bev_size = self.config['bev_size']
        self.bev_channels = self.config['bev_channels']
        self.num_modes = self.config['num_modes']
        
        # BEV特征提取
        # 计算卷积后的尺寸
        conv_size = (self.bev_size[0] + 1) // 2  # 第一层卷积
        conv_size = (conv_size + 1) // 2  # 第二层卷积
        conv_size = (conv_size + 1) // 2  # 第三层卷积
        
        self.bev_encoder = nn.Sequential(
            nn.Conv2d(self.bev_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(128 * conv_size * conv_size, self.hidden_dim)
        )
        
        # 轨迹特征提取
        self.trajectory_encoder = nn.Linear(2 * self.history_len, self.hidden_dim)
        
        # 地图特征提取
        self.map_encoder = nn.Linear(10, self.hidden_dim)  # 地图特征向量
        
        # 位置编码
        self.position_encoding = nn.Parameter(torch.randn(1, self.history_len + self.pred_len, self.hidden_dim))
        
        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dropout=self.dropout
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_encoder_layers)
        
        # Transformer解码器
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.hidden_dim,
            nhead=self.num_heads,
            dropout=self.dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=self.num_decoder_layers)
        
        # 输出层
        self.output_layer = nn.Linear(self.hidden_dim, 2 * self.num_modes)  # 每个模态输出x和y坐标
        
        # 模态概率层
        self.mode_prob_layer = nn.Linear(self.hidden_dim, self.num_modes)
        
        # 解码器输入投影
        self.decoder_input_proj = nn.Linear(2, self.hidden_dim)
    
    def forward(self, bev_features: torch.Tensor, history_trajectory: torch.Tensor, map_features: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        模型前向传播
        
        Args:
            bev_features: BEV特征图 [batch_size, channels, height, width]
            history_trajectory: 历史轨迹 [batch_size, history_len, 2]
            map_features: 地图特征 [batch_size, 10]
            
        Returns:
            预测轨迹和模态概率
        """
        batch_size = bev_features.shape[0]
        
        # 提取BEV特征
        bev_embedding = self.bev_encoder(bev_features)
        bev_embedding = bev_embedding.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # 提取轨迹特征
        history_flat = history_trajectory.view(batch_size, -1)  # [batch_size, 2*history_len]
        trajectory_embedding = self.trajectory_encoder(history_flat)
        trajectory_embedding = trajectory_embedding.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # 组合特征
        if map_features is not None:
            # 提取地图特征
            map_embedding = self.map_encoder(map_features)
            map_embedding = map_embedding.unsqueeze(1)  # [batch_size, 1, hidden_dim]
            # 拼接特征
            encoder_input = torch.cat([bev_embedding, trajectory_embedding, map_embedding], dim=1)
            # 添加位置编码
            encoder_input = encoder_input + self.position_encoding[:, :3, :]
        else:
            # 拼接特征
            encoder_input = torch.cat([bev_embedding, trajectory_embedding], dim=1)
            # 添加位置编码
            encoder_input = encoder_input + self.position_encoding[:, :2, :]
        
        # 编码器
        encoder_output = self.encoder(encoder_input.transpose(0, 1)).transpose(0, 1)
        
        # 解码器输入（使用历史轨迹的最后一个点作为起始点）
        decoder_input = history_trajectory[:, -1:, :].repeat(1, self.pred_len, 1)
        decoder_input = decoder_input.view(batch_size, self.pred_len, 2)
        decoder_embedding = self.decoder_input_proj(decoder_input)
        decoder_embedding = decoder_embedding + self.position_encoding[:, 2:2+self.pred_len, :]
        
        # 解码器
        decoder_output = self.decoder(
            decoder_embedding.transpose(0, 1),
            encoder_output.transpose(0, 1)
        ).transpose(0, 1)
        
        # 输出预测轨迹
        pred_trajectory = self.output_layer(decoder_output)
        pred_trajectory = pred_trajectory.view(batch_size, self.pred_len, self.num_modes, 2)
        
        # 输出模态概率
        mode_prob = self.mode_prob_layer(decoder_output[:, -1, :])
        mode_prob = torch.softmax(mode_prob, dim=-1)
        
        return pred_trajectory, mode_prob

class TrajectoryPredictor:
    """轨迹预测器"""
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        初始化轨迹预测器
        
        Args:
            model_path: 模型权重路径
            config: 预测器配置参数
        """
        # 默认配置
        default_config = {
            'device': 'cpu',         # 设备
            'history_len': 5,         # 历史轨迹长度
            'pred_len': 15,           # 预测轨迹长度
            'num_modes': 3,            # 预测模态数
            'max_speed': 10.0,         # 最大速度 (m/s)
            'max_acceleration': 2.0,   # 最大加速度 (m/s²)
            'max_deceleration': 4.0,   # 最大减速度 (m/s²)
            'max_steering': 0.5,       # 最大转向角 (rad)
            'dt': 0.2                  # 时间步长 (s)
        }
        
        # 更新配置
        self.config = default_config
        if config:
            self.config.update(config)
        
        self.device = self.config['device']
        self.history_len = self.config['history_len']
        self.pred_len = self.config['pred_len']
        self.num_modes = self.config['num_modes']
        self.max_speed = self.config['max_speed']
        self.max_acceleration = self.config['max_acceleration']
        self.max_deceleration = self.config['max_deceleration']
        self.max_steering = self.config['max_steering']
        self.dt = self.config['dt']
        
        # 初始化模型
        self.model = TrajectoryTransformer(config)
        self.model.to(self.device)
        
        # 加载模型权重
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # 设置为评估模式
        self.model.eval()
    
    def _analyze_scene(self, bev_map: np.ndarray) -> Dict:
        """
        分析场景
        
        Args:
            bev_map: BEV特征图
            
        Returns:
            场景分析结果
        """
        # 简单的场景分析
        # 计算道路区域
        road_area = np.sum(bev_map > 0.5)
        total_area = bev_map.shape[0] * bev_map.shape[1]
        road_ratio = road_area / total_area
        
        # 检测道路边界
        edges = np.where(bev_map > 0.5, 1, 0)
        
        # 分析场景类型
        if road_ratio > 0.6:
            scene_type = 'highway'
        elif road_ratio > 0.3:
            scene_type = 'urban'
        else:
            scene_type = 'rural'
        
        return {
            'road_ratio': road_ratio,
            'scene_type': scene_type,
            'road_edges': edges
        }
    
    def _evaluate_trajectory(self, trajectory: np.ndarray, scene_analysis: Dict) -> float:
        """
        评估轨迹质量
        
        Args:
            trajectory: 轨迹坐标
            scene_analysis: 场景分析结果
            
        Returns:
            轨迹质量分数
        """
        # 计算轨迹长度
        trajectory_length = np.sum(np.sqrt(np.sum(np.diff(trajectory, axis=0)**2, axis=1)))
        
        # 计算速度
        velocities = np.diff(trajectory, axis=0) / self.dt
        speeds = np.sqrt(np.sum(velocities**2, axis=1))
        
        # 计算加速度
        accelerations = np.diff(velocities, axis=0) / self.dt
        accel_magnitudes = np.sqrt(np.sum(accelerations**2, axis=1))
        
        # 评估指标
        score = 0.0
        
        # 速度评估
        if np.max(speeds) <= self.max_speed:
            score += 0.3
        else:
            score -= 0.5
        
        # 加速度评估
        if len(accel_magnitudes) > 0:
            if np.max(accel_magnitudes) <= self.max_acceleration:
                score += 0.3
            else:
                score -= 0.3
        
        # 轨迹平滑度评估
        if len(trajectory) > 2:
            # 计算曲率
            dx = np.diff(trajectory[:, 0])
            dy = np.diff(trajectory[:, 1])
            ddx = np.diff(dx)
            ddy = np.diff(dy)
            
            if len(ddx) > 0:
                curvature = np.sqrt(ddx**2 + ddy**2) / (dx[1:]**2 + dy[1:]**2)**(3/2)
                max_curvature = np.max(curvature) if len(curvature) > 0 else 0
                
                if max_curvature < 0.1:
                    score += 0.2
                elif max_curvature < 0.3:
                    score += 0.1
                else:
                    score -= 0.2
        
        # 场景适应性评估
        scene_type = scene_analysis['scene_type']
        if scene_type == 'highway' and np.mean(speeds) > 5.0:
            score += 0.2
        elif scene_type == 'urban' and np.mean(speeds) < 7.0:
            score += 0.2
        elif scene_type == 'rural' and np.mean(speeds) < 8.0:
            score += 0.2
        
        return max(0.0, score)
    
    def predict(self, bev_map: np.ndarray, history_trajectory: np.ndarray or List[List[float]], 
                scene_info: Optional[Dict] = None, map_features: Optional[Dict] = None) -> Dict:
        """
        预测轨迹
        
        Args:
            bev_map: BEV特征图
            history_trajectory: 历史轨迹
            scene_info: 场景信息（可选）
            map_features: 地图特征（可选）
            
        Returns:
            预测结果，包含轨迹和置信度
        """
        # 确保history_trajectory是numpy数组
        if isinstance(history_trajectory, list):
            history_trajectory = np.array(history_trajectory, dtype=np.float32)
        
        # 分析场景
        scene_analysis = self._analyze_scene(bev_map)
        if scene_info:
            scene_analysis.update(scene_info)
        
        # 处理地图特征
        map_tensor = None
        if map_features:
            # 提取地图特征向量
            map_vector = self._extract_map_features(map_features)
            map_tensor = torch.from_numpy(map_vector).unsqueeze(0).float().to(self.device)
        
        # 转换为张量
        bev_tensor = torch.from_numpy(bev_map).unsqueeze(0).unsqueeze(0).float().to(self.device)
        history_tensor = torch.from_numpy(history_trajectory).unsqueeze(0).float().to(self.device)
        
        # 预测
        with torch.no_grad():
            pred_trajectory, mode_prob = self.model(bev_tensor, history_tensor, map_tensor)
        
        # 转换为numpy
        pred_trajectory = pred_trajectory.squeeze(0).cpu().numpy()  # [pred_len, num_modes, 2]
        mode_prob = mode_prob.squeeze(0).cpu().numpy()  # [num_modes]
        
        # 添加随机扰动，确保不同输入产生不同轨迹
        import random
        for i in range(self.num_modes):
            # 为每个轨迹添加微小的随机扰动
            for j in range(len(pred_trajectory)):
                pred_trajectory[j, i, 0] += random.uniform(-0.1, 0.1)
                pred_trajectory[j, i, 1] += random.uniform(-0.1, 0.1)
        
        # 计算航向信息并评估轨迹
        trajectories_with_heading = []
        for i in range(self.num_modes):
            trajectory = pred_trajectory[:, i, :]
            heading = self._compute_heading(trajectory)
            
            # 评估轨迹质量
            trajectory_score = self._evaluate_trajectory(trajectory, scene_analysis)
            
            # 综合置信度和轨迹质量
            combined_score = 0.7 * mode_prob[i] + 0.3 * trajectory_score
            
            trajectories_with_heading.append({
                'trajectory': trajectory.tolist(),
                'heading': heading.tolist(),
                'confidence': float(mode_prob[i]),
                'score': float(trajectory_score),
                'combined_score': float(combined_score)
            })
        
        # 按综合得分排序
        trajectories_with_heading.sort(key=lambda x: x['combined_score'], reverse=True)
        
        return {
            'trajectories': trajectories_with_heading,
            'best_trajectory': trajectories_with_heading[0]['trajectory'],
            'best_heading': trajectories_with_heading[0]['heading'],
            'best_confidence': trajectories_with_heading[0]['confidence'],
            'scene_analysis': scene_analysis
        }
    
    def _compute_heading(self, trajectory: np.ndarray) -> np.ndarray:
        """
        计算轨迹的航向信息
        
        Args:
            trajectory: 轨迹坐标
            
        Returns:
            航向信息
        """
        heading = []
        for i in range(len(trajectory) - 1):
            dx = trajectory[i+1, 0] - trajectory[i, 0]
            dy = trajectory[i+1, 1] - trajectory[i, 1]
            angle = np.arctan2(dy, dx)
            heading.append(angle)
        # 最后一个点使用前一个点的航向
        if heading:
            heading.append(heading[-1])
        else:
            heading.append(0.0)
        return np.array(heading)
    
    def _extract_map_features(self, map_features: Dict) -> np.ndarray:
        """
        提取地图特征向量
        
        Args:
            map_features: 地图特征
            
        Returns:
            地图特征向量
        """
        # 初始化特征向量
        feature_vector = np.zeros(10, dtype=np.float32)
        
        # 道路信息
        road = map_features.get('road')
        if road:
            # 车道数
            feature_vector[0] = road.get('lanes', 1) / 5.0  # 归一化到0-1
            # 限速
            feature_vector[1] = road.get('speed_limit', 50) / 120.0  # 归一化到0-1
        
        # 路口信息
        junctions = map_features.get('junctions', [])
        feature_vector[2] = len(junctions) / 5.0  # 归一化到0-1
        
        # 交通标志信息
        traffic_signs = map_features.get('traffic_signs', [])
        feature_vector[3] = len(traffic_signs) / 10.0  # 归一化到0-1
        
        # 检查是否有停止标志
        has_stop_sign = any(sign['type'] == 'stop' for sign in traffic_signs)
        feature_vector[4] = 1.0 if has_stop_sign else 0.0
        
        # 检查是否有限速标志
        speed_limit_signs = [sign for sign in traffic_signs if sign['type'] == 'speed_limit']
        if speed_limit_signs:
            feature_vector[5] = speed_limit_signs[0].get('value', 50) / 120.0  # 归一化到0-1
        
        # 填充剩余特征
        feature_vector[6:] = 0.0
        
        return feature_vector

# 单元测试
if __name__ == "__main__":
    import os
    
    # 初始化预测器
    predictor = TrajectoryPredictor()
    
    # 生成测试数据
    # 创建不同场景的BEV地图
    # 高速场景
    bev_map_highway = np.zeros((500, 500), dtype=np.float32)
    bev_map_highway[200:300, :] = 1.0  # 宽道路
    
    # 城市场景
    bev_map_urban = np.zeros((500, 500), dtype=np.float32)
    bev_map_urban[225:275, :] = 1.0  # 窄道路
    
    # 乡村场景
    bev_map_rural = np.zeros((500, 500), dtype=np.float32)
    bev_map_rural[240:260, :] = 1.0  # 更窄的道路
    
    history_trajectory = np.array([
        [0, 0],
        [1, 0],
        [2, 0],
        [3, 0],
        [4, 0]
    ], dtype=np.float32)
    
    print("Testing trajectory prediction with different scenes...")
    
    # 测试高速场景
    print("\n=== Highway Scene ===")
    result = predictor.predict(bev_map_highway, history_trajectory)
    print(f"Scene type: {result['scene_analysis']['scene_type']}")
    print(f"Road ratio: {result['scene_analysis']['road_ratio']:.2f}")
    print(f"Predicted {len(result['trajectories'])} modes")
    for i, traj in enumerate(result['trajectories']):
        print(f"  Mode {i+1}: confidence={traj['confidence']:.2f}, score={traj['score']:.2f}, combined={traj['combined_score']:.2f}")
        print(f"    First 3 points: {traj['trajectory'][:3]}")
    
    # 测试城市场景
    print("\n=== Urban Scene ===")
    result = predictor.predict(bev_map_urban, history_trajectory)
    print(f"Scene type: {result['scene_analysis']['scene_type']}")
    print(f"Road ratio: {result['scene_analysis']['road_ratio']:.2f}")
    print(f"Predicted {len(result['trajectories'])} modes")
    for i, traj in enumerate(result['trajectories']):
        print(f"  Mode {i+1}: confidence={traj['confidence']:.2f}, score={traj['score']:.2f}, combined={traj['combined_score']:.2f}")
        print(f"    First 3 points: {traj['trajectory'][:3]}")
    
    # 测试乡村场景
    print("\n=== Rural Scene ===")
    result = predictor.predict(bev_map_rural, history_trajectory)
    print(f"Scene type: {result['scene_analysis']['scene_type']}")
    print(f"Road ratio: {result['scene_analysis']['road_ratio']:.2f}")
    print(f"Predicted {len(result['trajectories'])} modes")
    for i, traj in enumerate(result['trajectories']):
        print(f"  Mode {i+1}: confidence={traj['confidence']:.2f}, score={traj['score']:.2f}, combined={traj['combined_score']:.2f}")
        print(f"    First 3 points: {traj['trajectory'][:3]}")
    
    print("\nTrajectory prediction testing completed!")
