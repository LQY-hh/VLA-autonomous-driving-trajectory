import torch
try:
    from torch.quantization import quantize_dynamic
    from torch.quantization.quantize_fx import prepare_fx, convert_fx
    from torch.quantization import QConfigMapping
except ImportError:
    from torch.ao.quantization import quantize_dynamic
    from torch.ao.quantization.quantize_fx import prepare_fx, convert_fx
    from torch.ao.quantization import QConfigMapping
import numpy as np
from typing import Optional, Dict, Any

class ModelQuantizer:
    """
    模型量化类，用于减少模型大小和推理时间
    """
    
    def __init__(self):
        """
        初始化模型量化器
        """
        pass
    
    def dynamic_quantization(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        动态量化模型
        
        Args:
            model: 输入模型
            
        Returns:
            量化后的模型
        """
        try:
            # 对模型进行动态量化
            quantized_model = quantize_dynamic(
                model,
                {torch.nn.Linear, torch.nn.Conv2d, torch.nn.LSTM, torch.nn.GRU},
                dtype=torch.qint8
            )
            return quantized_model
        except Exception as e:
            print(f"Dynamic quantization failed: {e}")
            return model
    
    def static_quantization(self, model: torch.nn.Module, calibration_data: torch.Tensor) -> torch.nn.Module:
        """
        静态量化模型
        
        Args:
            model: 输入模型
            calibration_data: 用于校准的数据
            
        Returns:
            量化后的模型
        """
        try:
            # 设置模型为评估模式
            model.eval()
            
            # 准备量化
            qconfig = torch.quantization.get_default_qconfig('fbgemm')
            qconfig_mapping = QConfigMapping().set_global(qconfig)
            
            # 使用FX图模式进行量化
            prepared_model = prepare_fx(model, qconfig_mapping, (calibration_data,))
            
            # 校准模型
            with torch.no_grad():
                prepared_model(calibration_data)
            
            # 转换为量化模型
            quantized_model = convert_fx(prepared_model)
            return quantized_model
        except Exception as e:
            print(f"Static quantization failed: {e}")
            return model
    
    def int8_quantization(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        INT8量化模型
        
        Args:
            model: 输入模型
            
        Returns:
            量化后的模型
        """
        try:
            # 对模型进行INT8量化
            model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
            torch.quantization.prepare(model, inplace=True)
            # 这里应该使用实际数据进行校准
            # 为了演示，我们使用随机数据
            dummy_input = torch.randn(1, 1, 500, 500)
            model(dummy_input)
            torch.quantization.convert(model, inplace=True)
            return model
        except Exception as e:
            print(f"INT8 quantization failed: {e}")
            return model
    
    def optimize_for_edge(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        为边缘设备优化模型
        
        Args:
            model: 输入模型
            
        Returns:
            优化后的模型
        """
        try:
            # 结合多种优化技术
            # 1. 动态量化
            model = self.dynamic_quantization(model)
            
            # 2. 移动到CPU
            model = model.to('cpu')
            
            # 3. 设置为评估模式
            model.eval()
            
            return model
        except Exception as e:
            print(f"Edge optimization failed: {e}")
            return model
    
    def profile_quantization(self, model: torch.nn.Module, quantized_model: torch.nn.Module, input_data: torch.Tensor) -> Dict[str, Any]:
        """
        分析量化前后的性能差异
        
        Args:
            model: 原始模型
            quantized_model: 量化后的模型
            input_data: 输入数据
            
        Returns:
            性能分析结果
        """
        import time
        import os
        import psutil
        
        # 测量原始模型性能
        start_time = time.time()
        with torch.no_grad():
            original_output = model(input_data)
        original_time = time.time() - start_time
        
        # 测量量化模型性能
        start_time = time.time()
        with torch.no_grad():
            quantized_output = quantized_model(input_data)
        quantized_time = time.time() - start_time
        
        # 计算模型大小
        def get_model_size(model):
            torch.save(model.state_dict(), "temp_model.pt")
            size = os.path.getsize("temp_model.pt") / 1024 / 1024  # 转换为MB
            os.remove("temp_model.pt")
            return size
        
        original_size = get_model_size(model)
        quantized_size = get_model_size(quantized_model)
        
        # 测量内存使用
        process = psutil.Process(os.getpid())
        memory_usage = process.memory_info().rss / 1024 / 1024  # 转换为MB
        
        # 计算输出差异
        output_diff = None
        if isinstance(original_output, torch.Tensor) and isinstance(quantized_output, torch.Tensor):
            output_diff = torch.mean(torch.abs(original_output - quantized_output)).item()
        
        return {
            'original_inference_time': original_time,
            'quantized_inference_time': quantized_time,
            'speedup': original_time / quantized_time,
            'original_model_size_mb': original_size,
            'quantized_model_size_mb': quantized_size,
            'compression_ratio': original_size / quantized_size,
            'memory_usage_mb': memory_usage,
            'output_diff': output_diff
        }

if __name__ == "__main__":
    # 测试模型量化
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from trajectory.trajectory_predictor import TrajectoryTransformer
    
    # 创建模型
    model = TrajectoryTransformer()
    
    # 创建量化器
    quantizer = ModelQuantizer()
    
    # 动态量化
    dynamic_quantized_model = quantizer.dynamic_quantization(model)
    
    # 静态量化
    calibration_data = torch.randn(1, 1, 500, 500)
    static_quantized_model = quantizer.static_quantization(model, calibration_data)
    
    # 测试性能
    test_input = torch.randn(1, 1, 500, 500)
    test_history = torch.randn(1, 5, 2)
    
    # 为了测试，我们修改profile_quantization函数以适应TrajectoryTransformer的输入
    def profile_model(model, input_data, history):
        import time
        start_time = time.time()
        with torch.no_grad():
            output = model(input_data, history)
        end_time = time.time()
        return end_time - start_time
    
    original_time = profile_model(model, test_input, test_history)
    dynamic_time = profile_model(dynamic_quantized_model, test_input, test_history)
    static_time = profile_model(static_quantized_model, test_input, test_history)
    
    print(f"Original model inference time: {original_time:.4f} seconds")
    print(f"Dynamic quantized model inference time: {dynamic_time:.4f} seconds")
    print(f"Static quantized model inference time: {static_time:.4f} seconds")
    print(f"Dynamic speedup: {original_time / dynamic_time:.2f}x")
    print(f"Static speedup: {original_time / static_time:.2f}x")
