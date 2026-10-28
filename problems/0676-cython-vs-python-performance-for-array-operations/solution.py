import math
from typing import Dict, List, Any

def analyze_array_performance(n: int, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    分析并比较 Python、Cython 和 NumPy 在不同数组操作下的性能表现。
    
    Args:
        n: 数组大小（元素个数）
        operations: 操作列表，每个操作字典包含各方法的边际成本与固定开销 (ns)
        
    Returns:
        包含每项操作分析、Pipeline 总耗时、最佳推荐方案及 Crossover 临界值的字典。
    """
    ops_results = []
    crossover_sizes = {}
    
    # 管道总时间累计（未四舍五入的微秒数，保证精确度）
    pipeline_python_raw = 0.0
    pipeline_cython_raw = 0.0
    pipeline_numpy_raw = 0.0

    # 平局时的字典序优先级: cython < numpy < python
    tie_breaker = {"cython": 0, "numpy": 1, "python": 2}

    for op in operations:
        name = op['name']
        
        # 1. 计算各方法单次运行时间 (单位: 微秒 μs)
        py_us_raw = (op['python_overhead_ns'] + n * op['python_per_elem_ns']) / 1000.0
        cy_us_raw = (op['cython_overhead_ns'] + n * op['cython_per_elem_ns']) / 1000.0
        np_us_raw = (op['numpy_overhead_ns'] + n * op['numpy_per_elem_ns']) / 1000.0

        # 2. 累加至 Pipeline
        pipeline_python_raw += py_us_raw
        pipeline_cython_raw += cy_us_raw
        pipeline_numpy_raw += np_us_raw

        # 3. 计算相对 Python 的加速比 (Speedup)
        cy_speedup = round(py_us_raw / cy_us_raw, 2)
        np_speedup = round(py_us_raw / np_us_raw, 2)

        # 4. 判定单个 Operation 的最佳实现（按用时与字典序决断）
        candidates = [
            ('cython', cy_us_raw),
            ('numpy', np_us_raw),
            ('python', py_us_raw)
        ]
        best_method = min(candidates, key=lambda x: (x[1], tie_breaker[x[0]]))[0]

        # 5. 计算 NumPy 击败 Cython 的 Crossover Point
        cy_pe = op['cython_per_elem_ns']
        np_pe = op['numpy_per_elem_ns']
        cy_oh = op['cython_overhead_ns']
        np_oh = op['numpy_overhead_ns']

        if cy_pe <= np_pe:
            # Cython 边际成本更低或相等，NumPy 永远无法在大数组上赶超
            crossover = -1
        else:
            diff_pe = cy_pe - np_pe
            diff_oh = np_oh - cy_oh
            ratio = diff_oh / diff_pe
            
            if ratio <= 0:
                # 在 n=0 或小规模下 NumPy 开销已经不大于 Cython，起跑即超越
                crossover = 0
            else:
                # 向上取整得到使 NumPy 胜出的最小整数 n
                crossover = math.ceil(ratio)

        crossover_sizes[name] = crossover

        # 6. 打包单项操作结果（浮点数统一保留两位小数）
        ops_results.append({
            'name': name,
            'python_us': round(py_us_raw, 2),
            'cython_us': round(cy_us_raw, 2),
            'numpy_us': round(np_us_raw, 2),
            'cython_speedup': cy_speedup,
            'numpy_speedup': np_speedup,
            'best': best_method
        })

    # 7. 评估全局 Pipeline 最佳推荐方案
    pipeline_candidates = [
        ('cython', pipeline_cython_raw),
        ('numpy', pipeline_numpy_raw),
        ('python', pipeline_python_raw)
    ]
    recommended_method = min(pipeline_candidates, key=lambda x: (x[1], tie_breaker[x[0]]))[0]

    return {
        'operations': ops_results,
        'pipeline_python_us': round(pipeline_python_raw, 2),
        'pipeline_cython_us': round(pipeline_cython_raw, 2),
        'pipeline_numpy_us': round(pipeline_numpy_raw, 2),
        'recommended': recommended_method,
        'crossover_sizes': crossover_sizes
    }