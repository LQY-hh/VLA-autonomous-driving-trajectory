import os
import logging
import sys
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional
from pathlib import Path

class Logger:
    """日志工具类"""
    
    _instances = {}
    
    def __init__(self, name: str, log_dir: str = 'logs', level: int = logging.INFO):
        """
        初始化日志记录器
        
        Args:
            name: 日志记录器名称
            log_dir: 日志目录
            level: 日志级别
        """
        self.name = name
        self.log_dir = log_dir
        self.level = level
        
        if name not in Logger._instances:
            Logger._instances[name] = self._setup_logger()
    
    def _setup_logger(self) -> logging.Logger:
        """
        设置日志记录器
        
        Returns:
            配置好的日志记录器
        """
        logger = logging.getLogger(self.name)
        logger.setLevel(self.level)
        
        if logger.handlers:
            return logger
        
        os.makedirs(self.log_dir, exist_ok=True)
        
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.level)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        log_file = os.path.join(self.log_dir, f'{self.name}_{timestamp}.log')
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=10*1024*1024,
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(self.level)
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
        
        return logger
    
    @property
    def logger(self) -> logging.Logger:
        """
        获取日志记录器
        
        Returns:
            日志记录器实例
        """
        return Logger._instances.get(self.name)
    
    def debug(self, message: str):
        """记录调试信息"""
        self.logger.debug(message)
    
    def info(self, message: str):
        """记录一般信息"""
        self.logger.info(message)
    
    def warning(self, message: str):
        """记录警告信息"""
        self.logger.warning(message)
    
    def error(self, message: str, exc_info: bool = False):
        """记录错误信息"""
        self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message: str):
        """记录严重错误信息"""
        self.logger.critical(message)
    
    @staticmethod
    def get_logger(name: str = 'autonomous_driving') -> 'Logger':
        """
        获取日志记录器实例
        
        Args:
            name: 日志记录器名称
            
        Returns:
            Logger实例
        """
        if name not in Logger._instances:
            Logger._instances[name] = Logger(name)
        return Logger._instances[name]
    
    @staticmethod
    def cleanup_old_logs(log_dir: str = 'logs', days: int = 7):
        """
        清理过期的日志文件
        
        Args:
            log_dir: 日志目录
            days: 保留天数
        """
        if not os.path.exists(log_dir):
            return
        
        current_time = datetime.now().timestamp()
        cutoff_time = current_time - (days * 24 * 60 * 60)
        
        for file_name in os.listdir(log_dir):
            if not file_name.endswith('.log'):
                continue
            
            file_path = os.path.join(log_dir, file_name)
            file_time = os.path.getmtime(file_path)
            
            if file_time < cutoff_time:
                try:
                    os.remove(file_path)
                    print(f"Removed old log file: {file_path}")
                except Exception as e:
                    print(f"Error removing log file {file_path}: {e}")

def get_logger(name: str = 'autonomous_driving') -> Logger:
    """
    获取日志记录器的便捷函数
    
    Args:
        name: 日志记录器名称
        
    Returns:
        Logger实例
    """
    return Logger.get_logger(name)

# 单元测试
if __name__ == "__main__":
    logger = get_logger('test')
    
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")
    
    Logger.cleanup_old_logs(days=7)
