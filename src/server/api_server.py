import os
import sys
import json
import argparse
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import numpy as np
from io import BytesIO

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.cli.cli import AutonomousDrivingSystem

logger = logging.getLogger('api_server')

class TrajectoryPredictionHandler(BaseHTTPRequestHandler):
    """轨迹预测请求处理器"""
    
    system = None
    
    def do_GET(self):
        """处理GET请求"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/health':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'status': 'healthy', 'service': 'trajectory-prediction'}
            self.wfile.write(json.dumps(response).encode())
        
        elif parsed_path.path == '/status':
            self.send_response(200)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            if self.system:
                health = self.system.check_system_health()
                response = {'status': 'healthy' if health else 'unhealthy', 'modules': {}}
            else:
                response = {'status': 'not_initialized'}
            self.wfile.write(json.dumps(response).encode())
        
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'error': 'Not found'}
            self.wfile.write(json.dumps(response).encode())
    
    def do_POST(self):
        """处理POST请求"""
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/predict':
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                request = json.loads(post_data.decode('utf-8'))
                
                if 'image' not in request:
                    self.send_response(400)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    response = {'error': 'Missing image data'}
                    self.wfile.write(json.dumps(response).encode())
                    return
                
                image_data = request['image']
                
                result = self._process_prediction(image_data, request)
                
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps(result, ensure_ascii=False).encode())
            
            except Exception as e:
                self.send_response(500)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {'error': str(e)}
                self.wfile.write(json.dumps(response).encode())
        
        else:
            self.send_response(404)
            self.send_header('Content-type', 'application/json')
            self.end_headers()
            response = {'error': 'Not found'}
            self.wfile.write(json.dumps(response).encode())
    
    def _process_prediction(self, image_data, request):
        """
        处理预测请求
        
        Args:
            image_data: 图片数据（base64或文件路径）
            request: 请求参数
            
        Returns:
            预测结果
        """
        import cv2
        import base64
        
        if self.system is None:
            return {'error': 'System not initialized'}
        
        try:
            if isinstance(image_data, str):
                if os.path.exists(image_data):
                    image = cv2.imread(image_data)
                elif image_data.startswith('data:image'):
                    header, encoded = image_data.split(',', 1)
                    nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                else:
                    nparr = np.frombuffer(base64.b64decode(image_data), np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            else:
                image = image_data
            
            if image is None:
                return {'error': 'Failed to decode image'}
            
            result = self.system.predict(image)
            
            return result
        
        except Exception as e:
            return {'error': f'Prediction failed: {str(e)}'}
    
    def log_message(self, format, *args):
        """重写日志方法"""
        logger.info("%s - - [%s] %s" % (self.address_string(),
                                        self.log_date_time_string(),
                                        format % args))

def run_server(host='0.0.0.0', port=8000, multi_camera=False, map_path=None, multi_object=False):
    """
    运行API服务器
    
    Args:
        host: 主机地址
        port: 端口号
        multi_camera: 是否启用多摄像头
        map_path: 地图数据路径
        multi_object: 是否启用多目标预测
    """
    server_address = (host, port)
    httpd = HTTPServer(server_address, TrajectoryPredictionHandler)
    
    logger.info(f"Starting API server on {host}:{port}")
    print(f"Starting API server on {host}:{port}")
    
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped")
        print("\nServer stopped")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trajectory Prediction API Server')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Server host')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--multi-camera', action='store_true', help='Enable multi-camera processing')
    parser.add_argument('--map', type=str, help='Path to map data file')
    parser.add_argument('--multi-object', action='store_true', help='Enable multi-object trajectory prediction')
    
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    TrajectoryPredictionHandler.system = AutonomousDrivingSystem(
        multi_camera=args.multi_camera,
        map_path=args.map,
        multi_object=args.multi_object
    )
    
    run_server(args.host, args.port, args.multi_camera, args.map, args.multi_object)
