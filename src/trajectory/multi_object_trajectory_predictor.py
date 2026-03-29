import torch
import torch.nn as nn
import numpy as np
import os
from typing import Dict, List, Optional, Tuple

class MultiObjectTrajectoryTransformer(nn.Module):
    """
    多目标轨迹预测Transformer模型
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        初始化多目标轨迹预测模型
        
        Args:
            config: 模型配置参数
        """
        super(MultiObjectTrajectoryTransformer, self).__init__()
        
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
            'num_modes': 3,           # 预测模态数
            'max_objects': 10         # 最大目标数
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
        self.max_objects = self.config['max_objects']
        
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
        
        # 目标特征提取
        self.object_encoder = nn.Linear(2 * self.history_len + 2, self.hidden_dim)  # 2*history_len + 2 (类别和置信度)
        
        # 位置编码
        self.position_encoding = nn.Parameter(torch.randn(1, self.max_objects + 1, self.hidden_dim))  # +1 for BEV features
        
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
    
    def forward(self, bev_features: torch.Tensor, object_features: torch.Tensor, object_masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        模型前向传播
        
        Args:
            bev_features: BEV特征图 [batch_size, channels, height, width]
            object_features: 目标特征 [batch_size, max_objects, 2*history_len + 2]
            object_masks: 目标掩码 [batch_size, max_objects]
            
        Returns:
            预测轨迹和模态概率
        """
        batch_size = bev_features.shape[0]
        
        # 提取BEV特征
        bev_embedding = self.bev_encoder(bev_features)
        bev_embedding = bev_embedding.unsqueeze(1)  # [batch_size, 1, hidden_dim]
        
        # 提取目标特征
        object_embeddings = self.object_encoder(object_features)
        
        # 组合特征
        encoder_input = torch.cat([bev_embedding, object_embeddings], dim=1)  # [batch_size, 1 + max_objects, hidden_dim]
        
        # 添加位置编码
        encoder_input = encoder_input + self.position_encoding[:, :encoder_input.shape[1], :]
        
        # 创建注意力掩码
        encoder_mask = torch.zeros((batch_size, encoder_input.shape[1]), dtype=torch.bool, device=bev_features.device)
        encoder_mask[:, 0] = False  # BEV特征不被掩码
        encoder_mask[:, 1:] = object_masks  # 掩码无效目标
        
        # 编码器
        encoder_output = self.encoder(encoder_input.transpose(0, 1), src_key_padding_mask=encoder_mask).transpose(0, 1)
        
        # 准备解码器输入
        # 使用目标历史轨迹的最后一个点作为起始点
        object_history = object_features[:, :, :2*self.history_len].reshape(batch_size, self.max_objects, self.history_len, 2)
        last_points = object_history[:, :, -1, :]  # [batch_size, max_objects, 2]
        decoder_input = last_points.unsqueeze(2).repeat(1, 1, self.pred_len, 1)  # [batch_size, max_objects, pred_len, 2]
        decoder_input = decoder_input.reshape(batch_size, self.max_objects, self.pred_len, 2)
        
        # 预测结果
        pred_trajectories = []
        mode_probs = []
        
        # 为每个目标预测轨迹
        for i in range(self.max_objects):
            # 检查目标是否有效
            valid_mask = ~object_masks[:, i]
            if not torch.any(valid_mask):
                # 无效目标，添加零轨迹
                pred_trajectory = torch.zeros((batch_size, self.pred_len, self.num_modes, 2), device=bev_features.device)
                mode_prob = torch.zeros((batch_size, self.num_modes), device=bev_features.device)
            else:
                # 有效目标，进行预测
                obj_decoder_input = decoder_input[:, i, :, :]  # [batch_size, pred_len, 2]
                obj_decoder_embedding = self.decoder_input_proj(obj_decoder_input)
                
                # 解码器
                obj_decoder_output = self.decoder(
                    obj_decoder_embedding.transpose(0, 1),
                    encoder_output.transpose(0, 1)
                ).transpose(0, 1)
                
                # 输出预测轨迹
                pred_trajectory = self.output_layer(obj_decoder_output)
                pred_trajectory = pred_trajectory.view(batch_size, self.pred_len, self.num_modes, 2)
                
                # 输出模态概率
                mode_prob = self.mode_prob_layer(obj_decoder_output[:, -1, :])
                mode_prob = torch.softmax(mode_prob, dim=-1)
            
            pred_trajectories.append(pred_trajectory)
            mode_probs.append(mode_prob)
        
        # 拼接结果
        pred_trajectories = torch.stack(pred_trajectories, dim=1)  # [batch_size, max_objects, pred_len, num_modes, 2]
        mode_probs = torch.stack(mode_probs, dim=1)  # [batch_size, max_objects, num_modes]
        
        return pred_trajectories, mode_probs

class MultiObjectTrajectoryPredictor:
    """
    多目标轨迹预测器
    """
    
    def __init__(self, model_path: Optional[str] = None, config: Optional[Dict] = None):
        """
        初始化多目标轨迹预测器
        
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
            'max_objects': 10,         # 最大目标数
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
        self.max_objects = self.config['max_objects']
        self.max_speed = self.config['max_speed']
        self.max_acceleration = self.config['max_acceleration']
        self.max_deceleration = self.config['max_deceleration']
        self.max_steering = self.config['max_steering']
        self.dt = self.config['dt']
        
        # 初始化模型
        self.model = MultiObjectTrajectoryTransformer(config)
        self.model.to(self.device)
        
        # 加载模型权重
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        
        # 设置为评估模式
        self.model.eval()
    
    def predict(self, bev_map: np.ndarray, objects: List[Dict]) -> Dict:
        """
        预测多目标轨迹
        
        Args:
            bev_map: BEV特征图
            objects: 目标列表，每个目标包含历史轨迹、类别和置信度
            
        Returns:
            预测结果，包含每个目标的轨迹和置信度
        """
        # 准备目标特征
        object_features = np.zeros((self.max_objects, 2 * self.history_len + 2), dtype=np.float32)
        object_masks = np.ones(self.max_objects, dtype=bool)  # 1表示无效目标
        
        for i, obj in enumerate(objects[:self.max_objects]):
            # 历史轨迹
            history = obj.get('history', [])
            if len(history) >= self.history_len:
                # 使用最近的历史轨迹
                recent_history = history[-self.history_len:]
            else:
                # 补全历史轨迹
                recent_history = [[0, 0]] * (self.history_len - len(history)) + history
            
            # 转换为数组
            history_array = np.array(recent_history, dtype=np.float32).flatten()
            
            # 类别和置信度
            class_id = obj.get('class_id', 0)
            confidence = obj.get('confidence', 0.0)
            
            # 填充特征
            object_features[i, :2*self.history_len] = history_array
            object_features[i, 2*self.history_len] = class_id
            object_features[i, 2*self.history_len + 1] = confidence
            
            # 标记为有效目标
            object_masks[i] = False
        
        # 转换为张量
        bev_tensor = torch.from_numpy(bev_map).unsqueeze(0).unsqueeze(0).float().to(self.device)
        object_tensor = torch.from_numpy(object_features).unsqueeze(0).float().to(self.device)
        mask_tensor = torch.from_numpy(object_masks).unsqueeze(0).to(self.device)
        
        # 预测
        with torch.no_grad():
            pred_trajectories, mode_probs = self.model(bev_tensor, object_tensor, mask_tensor)
        
        # 转换为numpy
        pred_trajectories = pred_trajectories.squeeze(0).cpu().numpy()  # [max_objects, pred_len, num_modes, 2]
        mode_probs = mode_probs.squeeze(0).cpu().numpy()  # [max_objects, num_modes]
        
        # 处理预测结果
        results = []
        for i, obj in enumerate(objects[:self.max_objects]):
            if object_masks[i]:
                continue
            
            # 获取预测轨迹
            obj_trajectories = []
            for mode in range(self.num_modes):
                trajectory = pred_trajectories[i, :, mode, :]
                heading = self._compute_heading(trajectory)
                
                obj_trajectories.append({
                    'trajectory': trajectory.tolist(),
                    'heading': heading.tolist(),
                    'confidence': float(mode_probs[i, mode])
                })
            
            # 按置信度排序
            obj_trajectories.sort(key=lambda x: x['confidence'], reverse=True)
            
            results.append({
                'object_id': obj.get('id', i),
                'class_id': obj.get('class_id', 0),
                'class_name': obj.get('class_name', 'unknown'),
                'trajectories': obj_trajectories,
                'best_trajectory': obj_trajectories[0]['trajectory'],
                'best_heading': obj_trajectories[0]['heading'],
                'best_confidence': obj_trajectories[0]['confidence']
            })
        
        return {
            'objects': results,
            'num_objects': len(results)
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

if __name__ == "__main__":
    import os
    
    # 初始化预测器
    predictor = MultiObjectTrajectoryPredictor()
    
    # 生成测试数据
    bev_map = np.zeros((500, 500), dtype=np.float32)
    bev_map[225:275, :] = 1.0  # 道路
    
    # 模拟目标
    objects = [
        {
            'id': 0,
            'class_id': 1,
            'class_name': 'car',
            'confidence': 0.9,
            'history': [[10, 0], [11, 0], [12, 0], [13, 0], [14, 0]]
        },
        {
            'id': 1,
            'class_id': 2,
            'class_name': 'pedestrian',
            'confidence': 0.8,
            'history': [[20, 10], [21, 9], [22, 8], [23, 7], [24, 6]]
        }
    ]
    
    print("Testing multi-object trajectory prediction...")
    result = predictor.predict(bev_map, objects)
    
    print(f"Predicted trajectories for {result['num_objects']} objects")
    for obj in result['objects']:
        print(f"\nObject {obj['object_id']} ({obj['class_name']}):")
        print(f"  Best trajectory confidence: {obj['best_confidence']:.2f}")
        print(f"  First 3 points: {obj['best_trajectory'][:3]}")
        print(f"  First 3 headings: {obj['best_heading'][:3]}")
        
    print("\nMulti-object trajectory prediction testing completed!")
