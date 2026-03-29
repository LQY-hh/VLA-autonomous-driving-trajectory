import os
import torch
try:
    import onnx
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
from typing import Optional, Dict, Any

class ModelExporter:
    """模型导出工具"""
    
    def __init__(self, output_dir: str = 'models/exported'):
        """
        初始化模型导出器
        
        Args:
            output_dir: 导出目录路径
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def export_to_onnx(self, model: torch.nn.Module, input_sample: torch.Tensor, 
                      model_name: str = 'model', opset_version: int = 11) -> str:
        """
        将模型导出为ONNX格式
        
        Args:
            model: PyTorch模型
            input_sample: 输入样例
            model_name: 模型名称
            opset_version: ONNX opset版本
            
        Returns:
            导出的ONNX文件路径
        """
        if not ONNX_AVAILABLE:
            print("ONNX module not available. Skipping ONNX export.")
            return ""
        
        print(f"Exporting model to ONNX format...")
        
        # 设置模型为评估模式
        model.eval()
        
        # 导出路径
        onnx_path = os.path.join(self.output_dir, f'{model_name}.onnx')
        
        # 导出模型
        torch.onnx.export(
            model,
            input_sample,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )
        
        # 验证导出的模型
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print(f"ONNX model exported successfully: {onnx_path}")
        except Exception as e:
            print(f"Error validating ONNX model: {e}")
        
        return onnx_path
    
    def export_to_torchscript(self, model: torch.nn.Module, input_sample: torch.Tensor, 
                             model_name: str = 'model') -> str:
        """
        将模型导出为TorchScript格式
        
        Args:
            model: PyTorch模型
            input_sample: 输入样例
            model_name: 模型名称
            
        Returns:
            导出的TorchScript文件路径
        """
        print(f"Exporting model to TorchScript format...")
        
        # 设置模型为评估模式
        model.eval()
        
        # 导出路径
        ts_path = os.path.join(self.output_dir, f'{model_name}.pt')
        
        # 跟踪模型
        traced_model = torch.jit.trace(model, input_sample)
        
        # 保存模型
        torch.jit.save(traced_model, ts_path)
        
        print(f"TorchScript model exported successfully: {ts_path}")
        return ts_path
    
    def export_to_onnx_dynamic(self, model: torch.nn.Module, input_sample: torch.Tensor, 
                              model_name: str = 'model', opset_version: int = 11) -> str:
        """
        导出支持动态输入的ONNX模型
        
        Args:
            model: PyTorch模型
            input_sample: 输入样例
            model_name: 模型名称
            opset_version: ONNX opset版本
            
        Returns:
            导出的ONNX文件路径
        """
        if not ONNX_AVAILABLE:
            print("ONNX module not available. Skipping dynamic ONNX export.")
            return ""
        
        print(f"Exporting model to dynamic ONNX format...")
        
        # 设置模型为评估模式
        model.eval()
        
        # 导出路径
        onnx_path = os.path.join(self.output_dir, f'{model_name}_dynamic.onnx')
        
        # 导出模型
        torch.onnx.export(
            model,
            input_sample,
            onnx_path,
            export_params=True,
            opset_version=opset_version,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 1: 'sequence_length'},
                'output': {0: 'batch_size', 1: 'sequence_length'}
            }
        )
        
        # 验证导出的模型
        try:
            onnx_model = onnx.load(onnx_path)
            onnx.checker.check_model(onnx_model)
            print(f"Dynamic ONNX model exported successfully: {onnx_path}")
        except Exception as e:
            print(f"Error validating dynamic ONNX model: {e}")
        
        return onnx_path
    
    def optimize_onnx_model(self, onnx_path: str, optimized_model_name: str = 'model_optimized') -> str:
        """
        优化ONNX模型
        
        Args:
            onnx_path: ONNX模型路径
            optimized_model_name: 优化后模型名称
            
        Returns:
            优化后的ONNX文件路径
        """
        if not ONNX_AVAILABLE:
            print("ONNX module not available. Skipping ONNX optimization.")
            return onnx_path
        
        print(f"Optimizing ONNX model...")
        
        try:
            import onnxruntime as ort
            from onnxruntime.transformers import optimizer
            
            # 加载模型
            onnx_model = onnx.load(onnx_path)
            
            # 优化模型
            optimized_model = optimizer.optimize_model(
                onnx_model,
                model_type='bert',  # 根据模型类型调整
                num_heads=8,  # 根据模型配置调整
                hidden_size=512  # 根据模型配置调整
            )
            
            # 保存优化后的模型
            optimized_path = os.path.join(self.output_dir, f'{optimized_model_name}.onnx')
            onnx.save(optimized_model, optimized_path)
            
            print(f"Optimized ONNX model saved to: {optimized_path}")
            return optimized_path
        except Exception as e:
            print(f"Error optimizing ONNX model: {e}")
            return onnx_path
    
    def export_model(self, model: torch.nn.Module, input_sample: torch.Tensor, 
                    model_name: str = 'model', formats: list = ['onnx', 'torchscript']) -> Dict[str, str]:
        """
        导出模型到多种格式
        
        Args:
            model: PyTorch模型
            input_sample: 输入样例
            model_name: 模型名称
            formats: 导出格式列表
            
        Returns:
            导出文件路径字典
        """
        export_paths = {}
        
        if 'onnx' in formats:
            onnx_path = self.export_to_onnx(model, input_sample, model_name)
            export_paths['onnx'] = onnx_path
        
        if 'torchscript' in formats:
            ts_path = self.export_to_torchscript(model, input_sample, model_name)
            export_paths['torchscript'] = ts_path
        
        if 'onnx_dynamic' in formats:
            dynamic_onnx_path = self.export_to_onnx_dynamic(model, input_sample, model_name)
            export_paths['onnx_dynamic'] = dynamic_onnx_path
        
        return export_paths
    
    def get_model_info(self, model: torch.nn.Module) -> Dict[str, Any]:
        """
        获取模型信息
        
        Args:
            model: PyTorch模型
            
        Returns:
            模型信息字典
        """
        model_info = {
            'model_name': model.__class__.__name__,
            'parameters': sum(p.numel() for p in model.parameters()),
            'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad),
            'device': next(model.parameters()).device
        }
        
        print(f"Model info: {model_info}")
        return model_info

# 单元测试
if __name__ == "__main__":
    # 示例模型
    class SimpleModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = torch.nn.Linear(10, 50)
            self.fc2 = torch.nn.Linear(50, 1)
        
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # 初始化模型和导出器
    model = SimpleModel()
    exporter = ModelExporter()
    
    # 输入样例
    input_sample = torch.randn(1, 10)
    
    # 导出模型
    export_paths = exporter.export_model(model, input_sample, 'simple_model', ['onnx', 'torchscript', 'onnx_dynamic'])
    print(f"Exported model paths: {export_paths}")
    
    # 获取模型信息
    model_info = exporter.get_model_info(model)
    
    # 优化ONNX模型
    if 'onnx' in export_paths:
        optimized_path = exporter.optimize_onnx_model(export_paths['onnx'])
        print(f"Optimized model path: {optimized_path}")
