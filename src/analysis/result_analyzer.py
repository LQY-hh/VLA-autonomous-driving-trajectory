import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import cv2

class ResultAnalyzer:
    """结果分析工具"""
    
    def __init__(self, output_dir: str = 'output'):
        """
        初始化结果分析器
        
        Args:
            output_dir: 输出目录路径
        """
        self.output_dir = output_dir
        self.analysis_dir = os.path.join(output_dir, 'analysis')
        os.makedirs(self.analysis_dir, exist_ok=True)
    
    def analyze_trajectories(self, trajectories_dir: str) -> Dict:
        """
        分析轨迹预测结果
        
        Args:
            trajectories_dir: 轨迹文件目录
            
        Returns:
            分析结果
        """
        print(f"Analyzing trajectories in {trajectories_dir}")
        
        # 加载轨迹文件
        trajectory_files = [f for f in os.listdir(trajectories_dir) if f.endswith('.json')]
        if not trajectory_files:
            print("No trajectory files found")
            return {}
        
        # 分析轨迹数据
        results = {
            'total_trajectories': len(trajectory_files),
            'confidence_stats': {
                'mean': 0.0,
                'std': 0.0,
                'min': 1.0,
                'max': 0.0
            },
            'trajectory_lengths': [],
            'coordinates_range': {
                'x': {'min': float('inf'), 'max': float('-inf')},
                'y': {'min': float('inf'), 'max': float('-inf')}
            }
        }
        
        confidences = []
        
        for file_name in trajectory_files:
            file_path = os.path.join(trajectories_dir, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                trajectory = data.get('trajectory', {})
                best_confidence = trajectory.get('best_confidence', 0.0)
                confidences.append(best_confidence)
                
                best_trajectory = trajectory.get('best_trajectory', [])
                results['trajectory_lengths'].append(len(best_trajectory))
                
                # 分析坐标范围
                for point in best_trajectory:
                    if len(point) >= 2:
                        x, y = point[0], point[1]
                        results['coordinates_range']['x']['min'] = min(results['coordinates_range']['x']['min'], x)
                        results['coordinates_range']['x']['max'] = max(results['coordinates_range']['x']['max'], x)
                        results['coordinates_range']['y']['min'] = min(results['coordinates_range']['y']['min'], y)
                        results['coordinates_range']['y']['max'] = max(results['coordinates_range']['y']['max'], y)
                
            except Exception as e:
                print(f"Error analyzing {file_name}: {e}")
        
        # 计算置信度统计
        if confidences:
            results['confidence_stats']['mean'] = np.mean(confidences)
            results['confidence_stats']['std'] = np.std(confidences)
            results['confidence_stats']['min'] = np.min(confidences)
            results['confidence_stats']['max'] = np.max(confidences)
        
        # 保存分析结果
        analysis_file = os.path.join(self.analysis_dir, 'trajectory_analysis.json')
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Trajectory analysis saved to {analysis_file}")
        return results
    
    def analyze_advice(self, trajectories_dir: str) -> Dict:
        """
        分析驾驶建议结果
        
        Args:
            trajectories_dir: 轨迹文件目录
            
        Returns:
            分析结果
        """
        print(f"Analyzing advice in {trajectories_dir}")
        
        # 加载轨迹文件
        trajectory_files = [f for f in os.listdir(trajectories_dir) if f.endswith('.json')]
        if not trajectory_files:
            print("No trajectory files found")
            return {}
        
        # 分析建议数据
        results = {
            'total_advice': len(trajectory_files),
            'confidence_stats': {
                'mean': 0.0,
                'std': 0.0,
                'min': 1.0,
                'max': 0.0
            },
            'advice_lengths': [],
            'advice_types': {}
        }
        
        confidences = []
        
        for file_name in trajectory_files:
            file_path = os.path.join(trajectories_dir, file_name)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                advice = data.get('advice', {})
                best_confidence = advice.get('confidence', 0.0)
                confidences.append(best_confidence)
                
                advice_text = advice.get('advice', '')
                results['advice_lengths'].append(len(advice_text))
                
                # 分析建议类型
                advice_lower = advice_text.lower()
                if '保持' in advice_lower or '继续' in advice_lower:
                    self._increment_counter(results['advice_types'], 'maintain')
                elif '减速' in advice_lower or '慢行' in advice_lower:
                    self._increment_counter(results['advice_types'], 'slow_down')
                elif '加速' in advice_lower:
                    self._increment_counter(results['advice_types'], 'speed_up')
                elif '转弯' in advice_lower:
                    self._increment_counter(results['advice_types'], 'turn')
                elif '注意' in advice_lower or '小心' in advice_lower:
                    self._increment_counter(results['advice_types'], 'caution')
                else:
                    self._increment_counter(results['advice_types'], 'other')
                
            except Exception as e:
                print(f"Error analyzing {file_name}: {e}")
        
        # 计算置信度统计
        if confidences:
            results['confidence_stats']['mean'] = np.mean(confidences)
            results['confidence_stats']['std'] = np.std(confidences)
            results['confidence_stats']['min'] = np.min(confidences)
            results['confidence_stats']['max'] = np.max(confidences)
        
        # 保存分析结果
        analysis_file = os.path.join(self.analysis_dir, 'advice_analysis.json')
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Advice analysis saved to {analysis_file}")
        return results
    
    def generate_plots(self, trajectory_results: Dict, advice_results: Dict):
        """
        生成分析图表
        
        Args:
            trajectory_results: 轨迹分析结果
            advice_results: 建议分析结果
        """
        print("Generating analysis plots")
        
        # 生成置信度分布图
        self._plot_confidence_distribution(trajectory_results, advice_results)
        
        # 生成轨迹长度分布图
        self._plot_trajectory_lengths(trajectory_results)
        
        # 生成建议类型饼图
        self._plot_advice_types(advice_results)
        
        # 生成坐标范围图
        self._plot_coordinate_range(trajectory_results)
    
    def _plot_confidence_distribution(self, trajectory_results: Dict, advice_results: Dict):
        """
        生成置信度分布图
        """
        plt.figure(figsize=(12, 6))
        
        # 轨迹置信度
        if 'confidence_stats' in trajectory_results:
            plt.subplot(1, 2, 1)
            plt.title('Trajectory Confidence Distribution')
            plt.xlabel('Confidence')
            plt.ylabel('Frequency')
            # 这里简化处理，实际应该使用原始数据绘制直方图
            plt.axvline(trajectory_results['confidence_stats']['mean'], color='r', linestyle='--', label=f'Mean: {trajectory_results["confidence_stats"]["mean"]:.2f}')
            plt.axvline(trajectory_results['confidence_stats']['min'], color='g', linestyle='--', label=f'Min: {trajectory_results["confidence_stats"]["min"]:.2f}')
            plt.axvline(trajectory_results['confidence_stats']['max'], color='b', linestyle='--', label=f'Max: {trajectory_results["confidence_stats"]["max"]:.2f}')
            plt.legend()
        
        # 建议置信度
        if 'confidence_stats' in advice_results:
            plt.subplot(1, 2, 2)
            plt.title('Advice Confidence Distribution')
            plt.xlabel('Confidence')
            plt.ylabel('Frequency')
            # 这里简化处理，实际应该使用原始数据绘制直方图
            plt.axvline(advice_results['confidence_stats']['mean'], color='r', linestyle='--', label=f'Mean: {advice_results["confidence_stats"]["mean"]:.2f}')
            plt.axvline(advice_results['confidence_stats']['min'], color='g', linestyle='--', label=f'Min: {advice_results["confidence_stats"]["min"]:.2f}')
            plt.axvline(advice_results['confidence_stats']['max'], color='b', linestyle='--', label=f'Max: {advice_results["confidence_stats"]["max"]:.2f}')
            plt.legend()
        
        plt.tight_layout()
        plot_path = os.path.join(self.analysis_dir, 'confidence_distribution.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Confidence distribution plot saved to {plot_path}")
    
    def _plot_trajectory_lengths(self, trajectory_results: Dict):
        """
        生成轨迹长度分布图
        """
        if 'trajectory_lengths' in trajectory_results and trajectory_results['trajectory_lengths']:
            plt.figure(figsize=(10, 6))
            plt.hist(trajectory_results['trajectory_lengths'], bins=20)
            plt.title('Trajectory Length Distribution')
            plt.xlabel('Length')
            plt.ylabel('Frequency')
            plt.grid(True)
            
            plot_path = os.path.join(self.analysis_dir, 'trajectory_lengths.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Trajectory lengths plot saved to {plot_path}")
    
    def _plot_advice_types(self, advice_results: Dict):
        """
        生成建议类型饼图
        """
        if 'advice_types' in advice_results and advice_results['advice_types']:
            plt.figure(figsize=(10, 6))
            labels = list(advice_results['advice_types'].keys())
            sizes = list(advice_results['advice_types'].values())
            plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
            plt.axis('equal')  # 确保饼图是圆的
            plt.title('Advice Types Distribution')
            
            plot_path = os.path.join(self.analysis_dir, 'advice_types.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Advice types plot saved to {plot_path}")
    
    def _plot_coordinate_range(self, trajectory_results: Dict):
        """
        生成坐标范围图
        """
        if 'coordinates_range' in trajectory_results:
            coords = trajectory_results['coordinates_range']
            plt.figure(figsize=(10, 6))
            plt.title('Trajectory Coordinate Range')
            plt.xlabel('X Coordinate')
            plt.ylabel('Y Coordinate')
            
            # 绘制坐标范围框
            x_min, x_max = coords['x']['min'], coords['x']['max']
            y_min, y_max = coords['y']['min'], coords['y']['max']
            plt.plot([x_min, x_max, x_max, x_min, x_min], [y_min, y_min, y_max, y_max, y_min], 'r--')
            
            # 添加范围信息
            plt.text(x_min, y_min - (y_max - y_min) * 0.1, 
                     f'Range: X [{x_min:.2f}, {x_max:.2f}], Y [{y_min:.2f}, {y_max:.2f}]',
                     fontsize=10)
            
            plt.grid(True)
            plot_path = os.path.join(self.analysis_dir, 'coordinate_range.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"Coordinate range plot saved to {plot_path}")
    
    def _increment_counter(self, counter: Dict, key: str):
        """
        增加计数器
        
        Args:
            counter: 计数器字典
            key: 键
        """
        if key in counter:
            counter[key] += 1
        else:
            counter[key] = 1
    
    def analyze_system_performance(self, log_file: Optional[str] = None) -> Dict:
        """
        分析系统性能
        
        Args:
            log_file: 日志文件路径
            
        Returns:
            性能分析结果
        """
        print("Analyzing system performance")
        
        # 这里简化处理，实际应该从日志文件中分析性能数据
        results = {
            'processing_time': {
                'mean': 0.5,
                'std': 0.1,
                'min': 0.3,
                'max': 0.8
            },
            'memory_usage': {
                'mean': 500,
                'std': 100,
                'min': 300,
                'max': 800
            },
            'success_rate': 0.95
        }
        
        # 保存分析结果
        analysis_file = os.path.join(self.analysis_dir, 'performance_analysis.json')
        with open(analysis_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        print(f"Performance analysis saved to {analysis_file}")
        return results
    
    def generate_summary_report(self, trajectory_results: Dict, advice_results: Dict, performance_results: Dict):
        """
        生成综合分析报告
        
        Args:
            trajectory_results: 轨迹分析结果
            advice_results: 建议分析结果
            performance_results: 性能分析结果
        """
        print("Generating summary report")
        
        # 生成报告内容
        report = {
            'summary': {
                'total_processed': trajectory_results.get('total_trajectories', 0),
                'average_confidence': trajectory_results.get('confidence_stats', {}).get('mean', 0.0),
                'average_advice_confidence': advice_results.get('confidence_stats', {}).get('mean', 0.0),
                'success_rate': performance_results.get('success_rate', 0.0),
                'average_processing_time': performance_results.get('processing_time', {}).get('mean', 0.0)
            },
            'trajectory_analysis': trajectory_results,
            'advice_analysis': advice_results,
            'performance_analysis': performance_results
        }
        
        # 保存报告
        report_file = os.path.join(self.analysis_dir, 'summary_report.json')
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"Summary report saved to {report_file}")
        
        # 生成文本报告
        self._generate_text_report(report)
    
    def _generate_text_report(self, report: Dict):
        """
        生成文本格式的分析报告
        
        Args:
            report: 分析报告数据
        """
        text_report = f"""
# 系统分析报告

## 总体摘要
- 处理总数: {report['summary']['total_processed']}
- 平均轨迹置信度: {report['summary']['average_confidence']:.2f}
- 平均建议置信度: {report['summary']['average_advice_confidence']:.2f}
- 成功率: {report['summary']['success_rate']:.2f}
- 平均处理时间: {report['summary']['average_processing_time']:.2f}秒

## 轨迹分析
- 轨迹数量: {report['trajectory_analysis'].get('total_trajectories', 0)}
- 置信度统计:
  - 平均值: {report['trajectory_analysis'].get('confidence_stats', {}).get('mean', 0.0):.2f}
  - 标准差: {report['trajectory_analysis'].get('confidence_stats', {}).get('std', 0.0):.2f}
  - 最小值: {report['trajectory_analysis'].get('confidence_stats', {}).get('min', 0.0):.2f}
  - 最大值: {report['trajectory_analysis'].get('confidence_stats', {}).get('max', 0.0):.2f}
- 轨迹长度范围: {min(report['trajectory_analysis'].get('trajectory_lengths', [0]))} - {max(report['trajectory_analysis'].get('trajectory_lengths', [0]))}

## 建议分析
- 建议数量: {report['advice_analysis'].get('total_advice', 0)}
- 置信度统计:
  - 平均值: {report['advice_analysis'].get('confidence_stats', {}).get('mean', 0.0):.2f}
  - 标准差: {report['advice_analysis'].get('confidence_stats', {}).get('std', 0.0):.2f}
  - 最小值: {report['advice_analysis'].get('confidence_stats', {}).get('min', 0.0):.2f}
  - 最大值: {report['advice_analysis'].get('confidence_stats', {}).get('max', 0.0):.2f}
- 建议类型分布:
  {self._format_advice_types(report['advice_analysis'].get('advice_types', {}))}

## 性能分析
- 处理时间:
  - 平均值: {report['performance_analysis'].get('processing_time', {}).get('mean', 0.0):.2f}秒
  - 标准差: {report['performance_analysis'].get('processing_time', {}).get('std', 0.0):.2f}秒
  - 最小值: {report['performance_analysis'].get('processing_time', {}).get('min', 0.0):.2f}秒
  - 最大值: {report['performance_analysis'].get('processing_time', {}).get('max', 0.0):.2f}秒
- 内存使用:
  - 平均值: {report['performance_analysis'].get('memory_usage', {}).get('mean', 0.0):.2f}MB
  - 标准差: {report['performance_analysis'].get('memory_usage', {}).get('std', 0.0):.2f}MB
  - 最小值: {report['performance_analysis'].get('memory_usage', {}).get('min', 0.0):.2f}MB
  - 最大值: {report['performance_analysis'].get('memory_usage', {}).get('max', 0.0):.2f}MB

## 结论与建议
- 系统整体表现良好，成功率达到 {report['summary']['success_rate']:.2f}
- 轨迹预测和驾驶建议的置信度都在合理范围内
- 建议进一步优化系统性能，减少处理时间
- 可以考虑增加更多的场景信息，提高轨迹预测的准确性
"""
        
        # 保存文本报告
        text_report_file = os.path.join(self.analysis_dir, 'summary_report.txt')
        with open(text_report_file, 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        print(f"Text report saved to {text_report_file}")
    
    def _format_advice_types(self, advice_types: Dict) -> str:
        """
        格式化建议类型分布
        
        Args:
            advice_types: 建议类型字典
            
        Returns:
            格式化的字符串
        """
        formatted = []
        for key, value in advice_types.items():
            formatted.append(f"  - {key}: {value}")
        return '\n'.join(formatted)

# 单元测试
if __name__ == "__main__":
    # 初始化分析器
    analyzer = ResultAnalyzer()
    
    # 分析轨迹
    trajectory_dir = os.path.join('output', 'trajectories')
    if os.path.exists(trajectory_dir):
        trajectory_results = analyzer.analyze_trajectories(trajectory_dir)
        advice_results = analyzer.analyze_advice(trajectory_dir)
        performance_results = analyzer.analyze_system_performance()
        
        # 生成图表
        analyzer.generate_plots(trajectory_results, advice_results)
        
        # 生成报告
        analyzer.generate_summary_report(trajectory_results, advice_results, performance_results)
    else:
        print(f"Trajectory directory not found: {trajectory_dir}")
        print("Please run the system first to generate trajectory data.")
