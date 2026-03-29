import os
import time
import psutil
import logging
import json
from typing import Dict, Optional, List
import threading
from datetime import datetime

class SystemMonitor:
    """系统监控工具"""
    
    def __init__(self, log_dir: str = 'logs', monitor_interval: float = 1.0):
        """
        初始化系统监控器
        
        Args:
            log_dir: 日志目录路径
            monitor_interval: 监控间隔（秒）
        """
        self.log_dir = log_dir
        self.monitor_interval = monitor_interval
        os.makedirs(self.log_dir, exist_ok=True)
        
        # 配置日志
        self.logger = logging.getLogger('system_monitor')
        self.logger.setLevel(logging.INFO)
        
        # 控制台输出
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # 文件输出
        log_file = os.path.join(self.log_dir, 'system_monitor.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)
        
        # 监控数据
        self.monitoring_data = []
        self.is_monitoring = False
        self.monitor_thread = None
        
        # 性能指标阈值
        self.thresholds = {
            'cpu_usage': 80.0,  # CPU使用率阈值（%）
            'memory_usage': 80.0,  # 内存使用率阈值（%）
            'disk_usage': 80.0,  # 磁盘使用率阈值（%）
            'temperature': 70.0  # 温度阈值（°C）
        }
    
    def get_system_status(self) -> Dict:
        """
        获取系统状态
        
        Returns:
            系统状态字典
        """
        status = {
            'timestamp': datetime.now().isoformat(),
            'cpu': self._get_cpu_status(),
            'memory': self._get_memory_status(),
            'disk': self._get_disk_status(),
            'network': self._get_network_status(),
            'process': self._get_process_status(),
            'temperature': self._get_temperature()
        }
        return status
    
    def _get_cpu_status(self) -> Dict:
        """
        获取CPU状态
        """
        return {
            'usage_percent': psutil.cpu_percent(interval=0.1),
            'count': psutil.cpu_count(logical=True),
            'count_physical': psutil.cpu_count(logical=False),
            'freq': psutil.cpu_freq()._asdict()
        }
    
    def _get_memory_status(self) -> Dict:
        """
        获取内存状态
        """
        memory = psutil.virtual_memory()
        return {
            'total': memory.total,
            'available': memory.available,
            'used': memory.used,
            'percent': memory.percent
        }
    
    def _get_disk_status(self) -> Dict:
        """
        获取磁盘状态
        """
        disk = psutil.disk_usage('/')
        return {
            'total': disk.total,
            'used': disk.used,
            'free': disk.free,
            'percent': disk.percent
        }
    
    def _get_network_status(self) -> Dict:
        """
        获取网络状态
        """
        net_io = psutil.net_io_counters()
        return {
            'bytes_sent': net_io.bytes_sent,
            'bytes_recv': net_io.bytes_recv,
            'packets_sent': net_io.packets_sent,
            'packets_recv': net_io.packets_recv,
            'errin': net_io.errin,
            'errout': net_io.errout,
            'dropin': net_io.dropin,
            'dropout': net_io.dropout
        }
    
    def _get_process_status(self) -> Dict:
        """
        获取当前进程状态
        """
        process = psutil.Process(os.getpid())
        return {
            'pid': process.pid,
            'name': process.name(),
            'cpu_usage': process.cpu_percent(interval=0.1),
            'memory_usage': process.memory_info()._asdict(),
            'memory_percent': process.memory_percent(),
            'threads': process.num_threads()
        }
    
    def _get_temperature(self) -> Dict:
        """
        获取温度信息
        """
        try:
            # 尝试获取温度信息（仅在支持的系统上可用）
            if hasattr(psutil, 'sensors_temperatures'):
                temps = psutil.sensors_temperatures()
                return temps
            else:
                return {'error': 'Temperature sensors not available'}
        except Exception as e:
            return {'error': str(e)}
    
    def start_monitoring(self):
        """
        开始监控
        """
        if not self.is_monitoring:
            self.is_monitoring = True
            self.monitor_thread = threading.Thread(target=self._monitoring_loop)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            self.logger.info("System monitoring started")
    
    def stop_monitoring(self):
        """
        停止监控
        """
        if self.is_monitoring:
            self.is_monitoring = False
            if self.monitor_thread:
                self.monitor_thread.join(timeout=5.0)
            self.logger.info("System monitoring stopped")
    
    def _monitoring_loop(self):
        """
        监控循环
        """
        while self.is_monitoring:
            try:
                status = self.get_system_status()
                self.monitoring_data.append(status)
                
                # 检查阈值
                self._check_thresholds(status)
                
                # 保存监控数据
                self._save_monitoring_data()
                
                time.sleep(self.monitor_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.monitor_interval)
    
    def _check_thresholds(self, status: Dict):
        """
        检查阈值
        
        Args:
            status: 系统状态
        """
        # 检查CPU使用率
        cpu_usage = status['cpu']['usage_percent']
        if cpu_usage > self.thresholds['cpu_usage']:
            self.logger.warning(f"CPU usage exceeds threshold: {cpu_usage:.2f}%")
        
        # 检查内存使用率
        memory_usage = status['memory']['percent']
        if memory_usage > self.thresholds['memory_usage']:
            self.logger.warning(f"Memory usage exceeds threshold: {memory_usage:.2f}%")
        
        # 检查磁盘使用率
        disk_usage = status['disk']['percent']
        if disk_usage > self.thresholds['disk_usage']:
            self.logger.warning(f"Disk usage exceeds threshold: {disk_usage:.2f}%")
        
        # 检查温度
        temperature = status.get('temperature', {})
        if isinstance(temperature, dict) and 'coretemp' in temperature:
            for temp in temperature['coretemp']:
                if temp.current > self.thresholds['temperature']:
                    self.logger.warning(f"Temperature exceeds threshold: {temp.current:.2f}°C")
    
    def _save_monitoring_data(self):
        """
        保存监控数据
        """
        # 每10分钟保存一次数据
        if len(self.monitoring_data) >= 600 / self.monitor_interval:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            data_file = os.path.join(self.log_dir, f'monitoring_data_{timestamp}.json')
            
            with open(data_file, 'w', encoding='utf-8') as f:
                json.dump(self.monitoring_data, f, ensure_ascii=False, indent=2)
            
            self.logger.info(f"Monitoring data saved to {data_file}")
            # 清空数据
            self.monitoring_data = []
    
    def generate_report(self, report_file: Optional[str] = None) -> str:
        """
        生成监控报告
        
        Args:
            report_file: 报告文件路径
            
        Returns:
            报告文件路径
        """
        if not report_file:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            report_file = os.path.join(self.log_dir, f'monitoring_report_{timestamp}.json')
        
        # 获取当前系统状态
        current_status = self.get_system_status()
        
        # 生成报告
        report = {
            'report_time': datetime.now().isoformat(),
            'current_status': current_status,
            'thresholds': self.thresholds,
            'summary': self._generate_summary(current_status)
        }
        
        # 保存报告
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        self.logger.info(f"Monitoring report generated: {report_file}")
        return report_file
    
    def _generate_summary(self, status: Dict) -> Dict:
        """
        生成状态摘要
        
        Args:
            status: 系统状态
            
        Returns:
            状态摘要
        """
        summary = {
            'cpu_status': 'Normal' if status['cpu']['usage_percent'] < self.thresholds['cpu_usage'] else 'Warning',
            'memory_status': 'Normal' if status['memory']['percent'] < self.thresholds['memory_usage'] else 'Warning',
            'disk_status': 'Normal' if status['disk']['percent'] < self.thresholds['disk_usage'] else 'Warning',
            'overall_status': 'Normal'
        }
        
        # 检查是否有警告
        if summary['cpu_status'] == 'Warning' or summary['memory_status'] == 'Warning' or summary['disk_status'] == 'Warning':
            summary['overall_status'] = 'Warning'
        
        return summary
    
    def get_performance_metrics(self) -> Dict:
        """
        获取性能指标
        
        Returns:
            性能指标字典
        """
        status = self.get_system_status()
        return {
            'cpu_usage': status['cpu']['usage_percent'],
            'memory_usage': status['memory']['percent'],
            'disk_usage': status['disk']['percent'],
            'process_cpu': status['process']['cpu_usage'],
            'process_memory': status['process']['memory_percent']
        }

# 单元测试
if __name__ == "__main__":
    # 初始化监控器
    monitor = SystemMonitor(monitor_interval=2.0)
    
    # 开始监控
    monitor.start_monitoring()
    print("System monitoring started. Press Ctrl+C to stop.")
    
    try:
        # 运行一段时间
        for i in range(10):
            # 打印当前状态
            status = monitor.get_system_status()
            print(f"\nIteration {i+1}:")
            print(f"CPU Usage: {status['cpu']['usage_percent']:.2f}%")
            print(f"Memory Usage: {status['memory']['percent']:.2f}%")
            print(f"Disk Usage: {status['disk']['percent']:.2f}%")
            print(f"Process CPU: {status['process']['cpu_usage']:.2f}%")
            print(f"Process Memory: {status['process']['memory_percent']:.2f}%")
            
            time.sleep(2)
        
        # 生成报告
        report_file = monitor.generate_report()
        print(f"\nMonitoring report generated: {report_file}")
        
    except KeyboardInterrupt:
        print("\nStopping monitoring...")
    finally:
        # 停止监控
        monitor.stop_monitoring()
        print("System monitoring stopped.")
