import time
import torch
import numpy as np
from typing import Dict, Any, Optional

class PerformanceOptimizer:
    """
    性能优化类
    """
    
    def __init__(self):
        """
        初始化性能优化器
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.warmup_done = False
    
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """
        优化模型
        
        Args:
            model: 输入模型
            
        Returns:
            优化后的模型
        """
        # 移动模型到合适的设备
        model = model.to(self.device)
        
        # 设置为评估模式
        model.eval()
        
        # 进行warmup
        if not self.warmup_done:
            self._warmup(model)
            self.warmup_done = True
        
        # 尝试使用torch.jit加速
        try:
            model = torch.jit.script(model)
        except Exception as e:
            print(f"JIT scripting failed: {e}")
        
        # 尝试使用torch.jit.trace加速
        try:
            # 创建随机输入进行trace
            dummy_input = torch.randn(1, 1, 500, 500).to(self.device)
            dummy_history = torch.randn(1, 5, 2).to(self.device)
            model = torch.jit.trace(model, (dummy_input, dummy_history))
        except Exception as e:
            print(f"JIT tracing failed: {e}")
        
        return model
    
    def _warmup(self, model: torch.nn.Module):
        """
        模型warmup
        
        Args:
            model: 模型
        """
        try:
            # 创建随机输入进行warmup
            dummy_bev = torch.randn(1, 1, 500, 500).to(self.device)
            dummy_history = torch.randn(1, 5, 2).to(self.device)
            with torch.no_grad():
                for _ in range(5):
                    model(dummy_bev, dummy_history)
        except Exception as e:
            print(f"Warmup failed: {e}")
    
    @torch.no_grad()
    def optimize_inference(self, model: torch.nn.Module, input_data: torch.Tensor) -> torch.Tensor:
        """
        优化推理过程
        
        Args:
            model: 模型
            input_data: 输入数据
            
        Returns:
            推理结果
        """
        # 移动数据到设备
        input_data = input_data.to(self.device)
        
        # 使用torch.jit进行优化
        try:
            scripted_model = torch.jit.script(model)
            output = scripted_model(input_data)
        except Exception:
            # 如果脚本化失败，使用原始模型
            output = model(input_data)
        
        return output
    
    def batch_process(self, items: list, batch_size: int = 4) -> list:
        """
        批处理优化
        
        Args:
            items: 待处理的项目列表
            batch_size: 批处理大小
            
        Returns:
            处理结果列表
        """
        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i+batch_size]
            # 这里可以添加批处理逻辑
            results.extend(batch)
        return results
    
    def profile_inference(self, model: torch.nn.Module, input_data: torch.Tensor) -> Dict[str, Any]:
        """
        分析推理性能
        
        Args:
            model: 模型
            input_data: 输入数据
            
        Returns:
            性能分析结果
        """
        # 移动数据到设备
        input_data = input_data.to(self.device)
        
        # 预热
        for _ in range(3):
            with torch.no_grad():
                model(input_data)
        
        # 测量推理时间
        start_time = time.time()
        with torch.no_grad():
            output = model(input_data)
        end_time = time.time()
        
        inference_time = (end_time - start_time) * 1000  # 转换为毫秒
        
        # 测量内存使用
        memory_allocated = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        memory_reserved = torch.cuda.memory_reserved() / 1024 / 1024 if torch.cuda.is_available() else 0
        
        return {
            'inference_time_ms': inference_time,
            'memory_allocated_mb': memory_allocated,
            'memory_reserved_mb': memory_reserved,
            'output_shape': output.shape if hasattr(output, 'shape') else None
        }

class MemoryOptimizer:
    """
    内存优化类
    """
    
    @staticmethod
    def optimize_memory_usage():
        """
        优化内存使用
        """
        # 清理缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 清理Python垃圾回收
        import gc
        gc.collect()
    
    @staticmethod
    def reduce_batch_size(batch_size: int, memory_usage: float) -> int:
        """
        根据内存使用情况调整批处理大小
        
        Args:
            batch_size: 当前批处理大小
            memory_usage: 当前内存使用（MB）
            
        Returns:
            调整后的批处理大小
        """
        # 更智能的内存使用调整策略
        if memory_usage > 8000:  # 8GB
            return max(1, batch_size // 4)
        elif memory_usage > 6000:  # 6GB
            return max(1, batch_size // 3)
        elif memory_usage > 4000:  # 4GB
            return max(1, batch_size // 2)
        elif memory_usage > 2000:  # 2GB
            return max(1, batch_size)
        else:
            return batch_size
    
    @staticmethod
    def enable_gradient_checkpointing(model: torch.nn.Module):
        """
        启用梯度检查点，减少内存使用
        
        Args:
            model: 模型
        """
        try:
            from torch.utils.checkpoint import checkpoint_sequential, checkpoint
            # 对于不同类型的模型，启用不同的梯度检查点策略
            if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
                # 对于Transformer编码器
                model.encoder = torch.utils.checkpoint.checkpoint_sequential(
                    model.encoder.layers, 2, model.encoder
                )
            elif hasattr(model, 'layers'):
                # 对于具有layers属性的模型
                model.layers = torch.utils.checkpoint.checkpoint_sequential(
                    model.layers, 2, model
                )
        except Exception as e:
            print(f"Gradient checkpointing failed: {e}")
    
    @staticmethod
    def use_half_precision(model: torch.nn.Module):
        """
        使用半精度浮点数，减少内存使用
        
        Args:
            model: 模型
        """
        try:
            model = model.half()
        except Exception as e:
            print(f"Half precision conversion failed: {e}")
        return model
    
    @staticmethod
    def use_mixed_precision():
        """
        启用混合精度训练
        
        Returns:
            混合精度训练的GradScaler
        """
        try:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            return scaler
        except Exception as e:
            print(f"Mixed precision not available: {e}")
            return None
    
    @staticmethod
    def memory_aware_batch_size(data_loader, model, initial_batch_size=32, max_memory_usage=0.8):
        """
        基于内存使用情况自动调整批处理大小
        
        Args:
            data_loader: 数据加载器
            model: 模型
            initial_batch_size: 初始批处理大小
            max_memory_usage: 最大内存使用率
            
        Returns:
            优化后的批处理大小
        """
        import psutil
        import os
        
        # 获取系统内存总量
        total_memory = psutil.virtual_memory().total / 1024 / 1024  # MB
        max_memory = total_memory * max_memory_usage
        
        # 测试不同批处理大小的内存使用
        batch_size = initial_batch_size
        while batch_size > 1:
            try:
                # 创建测试批处理
                for batch in data_loader:
                    # 移动数据到设备
                    if isinstance(batch, (list, tuple)):
                        batch = [item.to(model.device) for item in batch]
                    else:
                        batch = batch.to(model.device)
                    
                    # 前向传播
                    with torch.no_grad():
                        output = model(batch)
                    
                    # 计算内存使用
                    memory_usage = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024
                    
                    if memory_usage < max_memory:
                        break
                    else:
                        batch_size = batch_size // 2
            except Exception as e:
                print(f"Error testing batch size: {e}")
                batch_size = batch_size // 2
        
        return batch_size
    
    @staticmethod
    def optimize_data_loading(prefetch_factor: int = 2, pin_memory: bool = True):
        """
        优化数据加载，减少内存使用
        
        Args:
            prefetch_factor: 预取因子
            pin_memory: 是否使用固定内存
            
        Returns:
            数据加载器配置
        """
        return {
            'prefetch_factor': prefetch_factor,
            'pin_memory': pin_memory
        }
    
    @staticmethod
    def clear_tensors(*tensors):
        """
        清理张量，释放内存
        
        Args:
            *tensors: 要清理的张量
        """
        for tensor in tensors:
            if tensor is not None:
                del tensor
        
        # 清理缓存
        MemoryOptimizer.optimize_memory_usage()

class ParallelProcessor:
    """
    并行处理器
    """
    
    def __init__(self, num_workers: int = None):
        """
        初始化并行处理器
        
        Args:
            num_workers: 工作线程数，默认使用CPU核心数
        """
        import os
        self.num_workers = num_workers or os.cpu_count()
        self.max_workers = min(self.num_workers, 16)  # 限制最大工作线程数
    
    def process_in_parallel(self, items: list, process_func, chunk_size: int = 1) -> list:
        """
        并行处理项目
        
        Args:
            items: 待处理的项目列表
            process_func: 处理函数
            chunk_size: 每个工作线程处理的项目数
            
        Returns:
            处理结果列表
        """
        from concurrent.futures import ThreadPoolExecutor
        
        if not items:
            return []
        
        # 动态调整工作线程数
        actual_workers = min(self.max_workers, len(items))
        
        results = []
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # 提交任务
            futures = [executor.submit(process_func, item) for item in items]
            
            # 处理结果
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Error processing item: {e}")
                    results.append(None)
        
        return results
    
    def process_in_parallel_batch(self, items: list, process_func, batch_size: int = 4, dynamic_batching: bool = True) -> list:
        """
        批处理并行处理项目
        
        Args:
            items: 待处理的项目列表
            process_func: 处理函数
            batch_size: 批处理大小
            dynamic_batching: 是否动态调整批处理大小
            
        Returns:
            处理结果列表
        """
        from concurrent.futures import ThreadPoolExecutor
        
        if not items:
            return []
        
        # 动态调整批处理大小
        if dynamic_batching:
            # 根据项目数量和工作线程数调整批处理大小
            optimal_batch_size = max(1, len(items) // self.max_workers)
            batch_size = min(batch_size, optimal_batch_size)
        
        # 分批次处理
        batches = [items[i:i+batch_size] for i in range(0, len(items), batch_size)]
        
        # 动态调整工作线程数
        actual_workers = min(self.max_workers, len(batches))
        
        results = []
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # 提交任务
            futures = [executor.submit(process_func, batch) for batch in batches]
            
            # 处理结果
            for future in futures:
                try:
                    batch_results = future.result()
                    results.extend(batch_results)
                except Exception as e:
                    print(f"Error processing batch: {e}")
                    # 为失败的批次添加None结果
                    batch_size = len(batches[0]) if batches else 0
                    results.extend([None] * batch_size)
        
        return results
    
    def process_with_multiprocessing(self, items: list, process_func, chunksize: int = 1) -> list:
        """
        使用多进程处理项目
        
        Args:
            items: 待处理的项目列表
            process_func: 处理函数
            chunksize: 每个进程处理的项目数
            
        Returns:
            处理结果列表
        """
        from multiprocessing import Pool
        
        if not items:
            return []
        
        # 动态调整进程数
        actual_processes = min(self.num_workers, len(items))
        
        results = []
        with Pool(processes=actual_processes) as pool:
            results = pool.map(process_func, items, chunksize=chunksize)
        
        return results
    
    def process_with_thread_pool_executor(self, items: list, process_func, max_workers: int = None) -> list:
        """
        使用ThreadPoolExecutor处理项目，支持取消任务
        
        Args:
            items: 待处理的项目列表
            process_func: 处理函数
            max_workers: 最大工作线程数
            
        Returns:
            处理结果列表
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        if not items:
            return []
        
        # 动态调整工作线程数
        actual_workers = max_workers or min(self.max_workers, len(items))
        
        results = []
        futures_dict = {}
        
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # 提交任务并保存future对象
            for i, item in enumerate(items):
                future = executor.submit(process_func, item)
                futures_dict[future] = i
            
            # 处理完成的任务
            for future in as_completed(futures_dict):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing item: {e}")
                    results.append(None)
        
        return results
    
    def parallel_map(self, func, *iterables, chunksize: int = 1) -> list:
        """
        并行版本的map函数
        
        Args:
            func: 处理函数
            *iterables: 可迭代对象
            chunksize: 每个工作线程处理的项目数
            
        Returns:
            处理结果列表
        """
        from concurrent.futures import ThreadPoolExecutor
        
        # 计算最小长度
        min_length = min(len(it) for it in iterables)
        if min_length == 0:
            return []
        
        # 动态调整工作线程数
        actual_workers = min(self.max_workers, min_length)
        
        results = []
        with ThreadPoolExecutor(max_workers=actual_workers) as executor:
            # 提交任务
            futures = [executor.submit(func, *args) for args in zip(*iterables)]
            
            # 处理结果
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as e:
                    print(f"Error processing item: {e}")
                    results.append(None)
        
        return results

if __name__ == "__main__":
    # 测试性能优化器
    optimizer = PerformanceOptimizer()
    
    # 创建测试模型
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from trajectory.trajectory_predictor import TrajectoryTransformer
    model = TrajectoryTransformer()
    
    # 优化模型
    optimized_model = optimizer.optimize_model(model)
    
    # 测试推理
    test_bev = torch.randn(1, 1, 500, 500)
    test_history = torch.randn(1, 5, 2)
    
    # 直接调用模型进行测试
    with torch.no_grad():
        result = model(test_bev, test_history)
    print(f"Inference result: trajectory shape={result[0].shape}, mode prob shape={result[1].shape}")
    
    # 分析性能
    # 创建性能分析函数
    def profile_model(model, bev, history):
        start_time = time.time()
        with torch.no_grad():
            output = model(bev, history)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000  # 转换为毫秒
        return {
            'inference_time_ms': inference_time,
            'output_shape': (output[0].shape, output[1].shape)
        }
    
    profile = profile_model(model, test_bev, test_history)
    print("Performance profile:")
    for key, value in profile.items():
        print(f"{key}: {value}")
    
    # 测试内存优化
    MemoryOptimizer.optimize_memory_usage()
    
    # 测试并行处理
    processor = ParallelProcessor()
    test_items = [1, 2, 3, 4, 5]
    def test_func(x):
        return x * 2
    
    parallel_results = processor.process_in_parallel(test_items, test_func)
    print(f"Parallel processing results: {parallel_results}")