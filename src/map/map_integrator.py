import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple

class MapIntegrator:
    """
    地图数据集成器
    """
    
    def __init__(self, map_data_path: Optional[str] = None):
        """
        初始化地图数据集成器
        
        Args:
            map_data_path: 地图数据路径
        """
        self.map_data_path = map_data_path
        self.map_data = {}
        
        # 加载地图数据
        if map_data_path and os.path.exists(map_data_path):
            self._load_map_data()
        else:
            # 使用默认地图数据
            self._load_default_map_data()
    
    def _load_map_data(self):
        """
        加载地图数据
        """
        try:
            with open(self.map_data_path, 'r', encoding='utf-8') as f:
                self.map_data = json.load(f)
            print(f"Loaded map data from {self.map_data_path}")
        except Exception as e:
            print(f"Error loading map data: {e}")
            # 使用默认地图数据
            self._load_default_map_data()
    
    def _load_default_map_data(self):
        """
        加载默认地图数据
        """
        # 默认地图数据
        self.map_data = {
            'roads': [
                {
                    'id': 'road_1',
                    'name': 'Main Street',
                    'lanes': 2,
                    'speed_limit': 50,  # km/h
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': [[0, 0], [100, 0], [200, 0]]
                    }
                },
                {
                    'id': 'road_2',
                    'name': 'Side Street',
                    'lanes': 1,
                    'speed_limit': 30,  # km/h
                    'geometry': {
                        'type': 'LineString',
                        'coordinates': [[50, 0], [50, 50], [50, 100]]
                    }
                }
            ],
            'junctions': [
                {
                    'id': 'junction_1',
                    'coordinates': [50, 0],
                    'connects': ['road_1', 'road_2']
                }
            ],
            'traffic_signs': [
                {
                    'id': 'sign_1',
                    'type': 'stop',
                    'coordinates': [45, 0]
                },
                {
                    'id': 'sign_2',
                    'type': 'speed_limit',
                    'value': 30,
                    'coordinates': [20, 0]
                }
            ]
        }
        print("Using default map data")
    
    def get_road_info(self, position: List[float]) -> Optional[Dict]:
        """
        获取指定位置的道路信息
        
        Args:
            position: 位置坐标 [x, y]
            
        Returns:
            道路信息，如果位置不在任何道路上则返回None
        """
        for road in self.map_data.get('roads', []):
            if self._is_point_on_road(position, road):
                return road
        return None
    
    def _is_point_on_road(self, point: List[float], road: Dict) -> bool:
        """
        检查点是否在道路上
        
        Args:
            point: 点坐标 [x, y]
            road: 道路信息
            
        Returns:
            点是否在道路上
        """
        # 简化实现：检查点是否在道路的 bounding box 内
        coordinates = road['geometry']['coordinates']
        x_coords = [coord[0] for coord in coordinates]
        y_coords = [coord[1] for coord in coordinates]
        
        min_x = min(x_coords)
        max_x = max(x_coords)
        min_y = min(y_coords)
        max_y = max(y_coords)
        
        # 添加一定的缓冲区
        buffer = 5.0
        return (min_x - buffer <= point[0] <= max_x + buffer and
                min_y - buffer <= point[1] <= max_y + buffer)
    
    def get_speed_limit(self, position: List[float]) -> float:
        """
        获取指定位置的限速
        
        Args:
            position: 位置坐标 [x, y]
            
        Returns:
            限速值（km/h）
        """
        # 先检查是否有交通标志
        for sign in self.map_data.get('traffic_signs', []):
            if sign['type'] == 'speed_limit':
                sign_pos = sign['coordinates']
                # 检查距离
                distance = np.sqrt((position[0] - sign_pos[0])**2 + (position[1] - sign_pos[1])**2)
                if distance < 10.0:
                    return sign.get('value', 50)
        
        # 检查道路限速
        road = self.get_road_info(position)
        if road:
            return road.get('speed_limit', 50)
        
        # 默认限速
        return 50
    
    def get_road_geometry(self, road_id: str) -> Optional[List[List[float]]]:
        """
        获取道路几何信息
        
        Args:
            road_id: 道路ID
            
        Returns:
            道路坐标列表
        """
        for road in self.map_data.get('roads', []):
            if road['id'] == road_id:
                return road['geometry']['coordinates']
        return None
    
    def get_junctions(self) -> List[Dict]:
        """
        获取所有路口信息
        
        Returns:
            路口信息列表
        """
        return self.map_data.get('junctions', [])
    
    def get_traffic_signs(self, area: Optional[Dict[str, float]] = None) -> List[Dict]:
        """
        获取指定区域内的交通标志
        
        Args:
            area: 区域边界，格式为 {'min_x': float, 'max_x': float, 'min_y': float, 'max_y': float}
            
        Returns:
            交通标志列表
        """
        if not area:
            return self.map_data.get('traffic_signs', [])
        
        signs_in_area = []
        for sign in self.map_data.get('traffic_signs', []):
            x, y = sign['coordinates']
            if (area['min_x'] <= x <= area['max_x'] and
                area['min_y'] <= y <= area['max_y']):
                signs_in_area.append(sign)
        
        return signs_in_area
    
    def integrate_with_bev(self, bev_map: np.ndarray, vehicle_position: List[float]) -> np.ndarray:
        """
        将地图信息集成到BEV中
        
        Args:
            bev_map: BEV特征图
            vehicle_position: 车辆位置
            
        Returns:
            集成了地图信息的BEV特征图
        """
        # 复制BEV图
        integrated_bev = bev_map.copy()
        
        # 获取车辆周围的道路信息
        road = self.get_road_info(vehicle_position)
        if road:
            # 绘制道路
            self._draw_road(integrated_bev, road, vehicle_position)
        
        # 获取车辆周围的交通标志
        area = {
            'min_x': vehicle_position[0] - 50,
            'max_x': vehicle_position[0] + 50,
            'min_y': vehicle_position[1] - 50,
            'max_y': vehicle_position[1] + 50
        }
        signs = self.get_traffic_signs(area)
        for sign in signs:
            # 绘制交通标志
            self._draw_traffic_sign(integrated_bev, sign, vehicle_position)
        
        return integrated_bev
    
    def _draw_road(self, bev_map: np.ndarray, road: Dict, vehicle_position: List[float]):
        """
        在BEV图上绘制道路
        
        Args:
            bev_map: BEV特征图
            road: 道路信息
            vehicle_position: 车辆位置
        """
        import cv2
        
        # 转换道路坐标到BEV坐标
        bev_coords = []
        for coord in road['geometry']['coordinates']:
            # 简化的坐标转换：假设车辆在BEV中心
            bev_x = int(bev_map.shape[1] // 2 + (coord[0] - vehicle_position[0]))
            bev_y = int(bev_map.shape[0] // 2 - (coord[1] - vehicle_position[1]))
            bev_coords.append([bev_x, bev_y])
        
        # 绘制道路
        if len(bev_coords) > 1:
            for i in range(len(bev_coords) - 1):
                cv2.line(bev_map, tuple(bev_coords[i]), tuple(bev_coords[i+1]), 0.7, 3)
        
        # 绘制车道线
        lanes = road.get('lanes', 1)
        if lanes > 1:
            lane_width = 3  # 车道宽度（米）
            for i in range(1, lanes):
                offset = lane_width * (i - lanes/2 + 0.5)
                lane_coords = []
                for coord in road['geometry']['coordinates']:
                    # 计算车道线坐标
                    # 简化实现：假设道路是直线
                    if len(road['geometry']['coordinates']) >= 2:
                        start = road['geometry']['coordinates'][0]
                        end = road['geometry']['coordinates'][-1]
                        # 计算道路方向向量
                        dx = end[0] - start[0]
                        dy = end[1] - start[1]
                        length = np.sqrt(dx**2 + dy**2)
                        if length > 0:
                            # 计算垂直方向向量
                            perp_dx = -dy / length
                            perp_dy = dx / length
                            # 计算车道线点
                            lane_x = coord[0] + perp_dx * offset
                            lane_y = coord[1] + perp_dy * offset
                            # 转换到BEV坐标
                            bev_x = int(bev_map.shape[1] // 2 + (lane_x - vehicle_position[0]))
                            bev_y = int(bev_map.shape[0] // 2 - (lane_y - vehicle_position[1]))
                            lane_coords.append([bev_x, bev_y])
                
                # 绘制车道线
                if len(lane_coords) > 1:
                    for j in range(len(lane_coords) - 1):
                        cv2.line(bev_map, tuple(lane_coords[j]), tuple(lane_coords[j+1]), 0.5, 1)
    
    def _draw_traffic_sign(self, bev_map: np.ndarray, sign: Dict, vehicle_position: List[float]):
        """
        在BEV图上绘制交通标志
        
        Args:
            bev_map: BEV特征图
            sign: 交通标志信息
            vehicle_position: 车辆位置
        """
        import cv2
        
        # 转换标志坐标到BEV坐标
        sign_x, sign_y = sign['coordinates']
        bev_x = int(bev_map.shape[1] // 2 + (sign_x - vehicle_position[0]))
        bev_y = int(bev_map.shape[0] // 2 - (sign_y - vehicle_position[1]))
        
        # 检查坐标是否在BEV范围内
        if 0 <= bev_x < bev_map.shape[1] and 0 <= bev_y < bev_map.shape[0]:
            # 绘制标志
            if sign['type'] == 'stop':
                # 绘制停止标志（八边形）
                cv2.circle(bev_map, (bev_x, bev_y), 8, 0.9, -1)
            elif sign['type'] == 'speed_limit':
                # 绘制限速标志（圆形）
                cv2.circle(bev_map, (bev_x, bev_y), 6, 0.8, -1)
                # 绘制限速值
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(bev_map, str(sign.get('value', '')), 
                           (bev_x - 4, bev_y + 2), font, 0.3, 0, 1)
    
    def get_map_features(self, position: List[float], radius: float = 50.0) -> Dict:
        """
        获取指定位置周围的地图特征
        
        Args:
            position: 位置坐标 [x, y]
            radius: 搜索半径
            
        Returns:
            地图特征
        """
        features = {
            'road': None,
            'speed_limit': self.get_speed_limit(position),
            'junctions': [],
            'traffic_signs': []
        }
        
        # 获取道路信息
        features['road'] = self.get_road_info(position)
        
        # 获取路口信息
        for junction in self.map_data.get('junctions', []):
            jx, jy = junction['coordinates']
            distance = np.sqrt((position[0] - jx)**2 + (position[1] - jy)**2)
            if distance <= radius:
                features['junctions'].append(junction)
        
        # 获取交通标志
        area = {
            'min_x': position[0] - radius,
            'max_x': position[0] + radius,
            'min_y': position[1] - radius,
            'max_y': position[1] + radius
        }
        features['traffic_signs'] = self.get_traffic_signs(area)
        
        return features

if __name__ == "__main__":
    # 测试地图集成器
    map_integrator = MapIntegrator()
    
    # 测试获取道路信息
    position = [10, 0]
    road_info = map_integrator.get_road_info(position)
    print(f"Road info at {position}: {road_info}")
    
    # 测试获取限速
    speed_limit = map_integrator.get_speed_limit(position)
    print(f"Speed limit at {position}: {speed_limit}")
    
    # 测试获取地图特征
    features = map_integrator.get_map_features(position)
    print(f"Map features at {position}: {features}")
    
    # 测试BEV集成
    bev_map = np.zeros((500, 500), dtype=np.float32)
    integrated_bev = map_integrator.integrate_with_bev(bev_map, position)
    
    # 可视化
    import cv2
    bev_visual = (integrated_bev * 255).astype(np.uint8)
    bev_visual = cv2.cvtColor(bev_visual, cv2.COLOR_GRAY2BGR)
    
    # 保存结果
    output_path = "map_integrated_bev.jpg"
    cv2.imwrite(output_path, bev_visual)
    print(f"Map integrated BEV saved to {output_path}")
