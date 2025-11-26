#!/usr/bin/env python3
"""
简化的 ONNX 模型转换脚本
支持转换为 MindSpore、PaddlePaddle 和 JAX 格式
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

BASE_DIR = "/mnt/public/users/chenyitong/projects/SliMM"

# 模型映射
MODELS = {
    'A': ('slimm_onnx/SliMM_A.onnx', 'slimm_mindspore', 'slimm_paddle', 'slimm_jax'),
    'B': ('slimm_onnx/SliMM_B.onnx', 'slimm_mindspore', 'slimm_paddle', 'slimm_jax'),
    'C': ('slimm_onnx/SliMM_C.onnx', 'slimm_mindspore', 'slimm_paddle', 'slimm_jax'),
    'D': ('slimm_onnx/SliMM_D.onnx', 'slimm_mindspore', 'slimm_paddle', 'slimm_jax'),
    'E': ('slimm_onnx/SliMM_E.onnx', 'slimm_mindspore', 'slimm_paddle', 'slimm_jax'),
    'F': ('slimm_onnx_2/SliMM_F.onnx', 'slimm_mindspore_2', 'slimm_paddle_2', 'slimm_jax_2'),
}


def check_dependencies():
    """检查所需的依赖包"""
    print("Checking dependencies...")
    dependencies = {
        'mindspore': False,
        'paddlepaddle': False,
        'jax': False,
        'onnx': False,
        'onnxruntime': False,
    }
    
    # 检查 MindSpore
    try:
        import mindspore
        dependencies['mindspore'] = True
        print("  ✓ MindSpore installed")
    except ImportError:
        print("  ✗ MindSpore not installed (pip install mindspore)")
    
    # 检查 PaddlePaddle
    try:
        import paddle
        dependencies['paddlepaddle'] = True
        print("  ✓ PaddlePaddle installed")
    except ImportError:
        print("  ✗ PaddlePaddle not installed (pip install paddlepaddle)")
    
    # 检查 JAX
    try:
        import jax
        dependencies['jax'] = True
        print("  ✓ JAX installed")
    except ImportError:
        print("  ✗ JAX not installed (pip install jax jaxlib)")
    
    # 检查 ONNX (可选，我们主要用 onnxruntime)
    try:
        import onnx
        dependencies['onnx'] = True
        print("  ✓ ONNX installed")
    except (ImportError, Exception):
        print("  ⚠ ONNX not available (optional, using onnxruntime instead)")
    
    # 检查 ONNX Runtime
    try:
        import onnxruntime
        dependencies['onnxruntime'] = True
        print("  ✓ ONNX Runtime installed")
    except ImportError:
        print("  ✗ ONNX Runtime not installed (pip install onnxruntime)")
    
    return dependencies


def convert_to_mindspore_simple(onnx_path, output_dir, model_name):
    """使用简单方法转换为 MindSpore（创建包装器）"""
    print(f"\nConverting {model_name} to MindSpore format...")
    
    try:
        import mindspore as ms
        import onnxruntime as ort
        import numpy as np
        import pickle
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载 ONNX 模型获取信息
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # 创建一个包装器类
        class ONNXWrapper:
            def __init__(self, onnx_path):
                self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                self.input_names = [inp.name for inp in self.session.get_inputs()]
                self.output_names = [out.name for out in self.session.get_outputs()]
            
            def __call__(self, *args):
                inputs = {}
                for i, arg in enumerate(args):
                    if i < len(self.input_names):
                        # 将 MindSpore Tensor 转换为 numpy
                        if hasattr(arg, 'asnumpy'):
                            inputs[self.input_names[i]] = arg.asnumpy()
                        else:
                            inputs[self.input_names[i]] = np.array(arg)
                
                outputs = self.session.run(None, inputs)
                
                # 转换回 MindSpore Tensor
                if len(outputs) == 1:
                    return ms.Tensor(outputs[0], dtype=ms.float32)
                else:
                    return [ms.Tensor(out, dtype=ms.float32) for out in outputs]
        
        wrapper = ONNXWrapper(onnx_path)
        
        # 保存包装器
        output_file = os.path.join(output_dir, f"{model_name}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(wrapper, f)
        
        print(f"  ✓ Created MindSpore wrapper: {output_file}")
        print(f"    Note: This is a wrapper around ONNX Runtime.")
        print(f"    For native MindSpore, use mindconverter tool.")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def convert_to_paddlepaddle_simple(onnx_path, output_dir, model_name):
    """使用简单方法转换为 PaddlePaddle（创建包装器）"""
    print(f"\nConverting {model_name} to PaddlePaddle format...")
    
    try:
        import paddle
        import onnxruntime as ort
        import numpy as np
        import pickle
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载 ONNX 模型获取信息
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # 创建一个包装器类
        class ONNXWrapper:
            def __init__(self, onnx_path):
                self.session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
                self.input_names = [inp.name for inp in self.session.get_inputs()]
                self.output_names = [out.name for out in self.session.get_outputs()]
            
            def __call__(self, *args):
                inputs = {}
                for i, arg in enumerate(args):
                    if i < len(self.input_names):
                        # 将 PaddlePaddle Tensor 转换为 numpy
                        if hasattr(arg, 'numpy'):
                            inputs[self.input_names[i]] = arg.numpy()
                        else:
                            inputs[self.input_names[i]] = np.array(arg)
                
                outputs = self.session.run(None, inputs)
                
                # 转换回 PaddlePaddle Tensor
                if len(outputs) == 1:
                    return paddle.to_tensor(outputs[0], dtype='float32')
                else:
                    return [paddle.to_tensor(out, dtype='float32') for out in outputs]
        
        wrapper = ONNXWrapper(onnx_path)
        
        # 保存包装器
        output_file = os.path.join(output_dir, f"{model_name}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(wrapper, f)
        
        print(f"  ✓ Created PaddlePaddle wrapper: {output_file}")
        print(f"    Note: This is a wrapper around ONNX Runtime.")
        print(f"    For native PaddlePaddle, use x2paddle tool.")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def convert_to_jax_simple(onnx_path, output_dir, model_name):
    """使用简单方法转换为 JAX（创建包装器）"""
    print(f"\nConverting {model_name} to JAX format...")
    
    try:
        import jax
        import jax.numpy as jnp
        import onnxruntime as ort
        import numpy as np
        import pickle
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载 ONNX 模型获取信息
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        
        # 创建一个包装器函数
        def create_jax_wrapper(onnx_path):
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            input_names = [inp.name for inp in session.get_inputs()]
            output_names = [out.name for out in session.get_outputs()]
            
            def jax_forward(*args):
                inputs = {}
                for i, arg in enumerate(args):
                    if i < len(input_names):
                        # 将 JAX 数组转换为 numpy
                        inputs[input_names[i]] = np.array(arg)
                
                outputs = session.run(None, inputs)
                
                # 转换回 JAX 数组
                if len(outputs) == 1:
                    return jnp.array(outputs[0])
                else:
                    return [jnp.array(out) for out in outputs]
            
            return jax_forward
        
        wrapper = create_jax_wrapper(onnx_path)
        
        # 保存包装器
        output_file = os.path.join(output_dir, f"{model_name}.pkl")
        with open(output_file, 'wb') as f:
            pickle.dump(wrapper, f)
        
        print(f"  ✓ Created JAX wrapper: {output_file}")
        print(f"    Note: This is a wrapper around ONNX Runtime.")
        return True
        
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False


def main():
    """主函数"""
    print("="*70)
    print("ONNX Model Converter to MindSpore/PaddlePaddle/JAX")
    print("="*70)
    
    # 检查依赖
    deps = check_dependencies()
    
    # 检查 ONNX 模型文件
    print("\nChecking ONNX model files...")
    missing = []
    for model_name, (onnx_path, _, _, _) in MODELS.items():
        full_path = os.path.join(BASE_DIR, onnx_path)
        if os.path.exists(full_path):
            print(f"  ✓ {model_name}: {onnx_path}")
        else:
            print(f"  ✗ {model_name}: {onnx_path} (NOT FOUND)")
            missing.append(model_name)
    
    if missing:
        print(f"\nWarning: {len(missing)} model(s) not found. Please check the paths.")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return
    
    # 转换模型
    print("\n" + "="*70)
    print("Starting conversion...")
    print("="*70)
    
    results = {'mindspore': 0, 'paddlepaddle': 0, 'jax': 0}
    
    for model_name, (onnx_path, ms_dir, pd_dir, jax_dir) in MODELS.items():
        full_onnx_path = os.path.join(BASE_DIR, onnx_path)
        if not os.path.exists(full_onnx_path):
            continue
        
        model_file = f"SliMM_{model_name}"
        
        print(f"\n{'='*70}")
        print(f"Processing: {model_file}")
        print(f"{'='*70}")
        
        # 转换为 MindSpore
        ms_output_dir = os.path.join(BASE_DIR, ms_dir)
        if convert_to_mindspore_simple(full_onnx_path, ms_output_dir, model_file):
            results['mindspore'] += 1
        
        # 转换为 PaddlePaddle
        pd_output_dir = os.path.join(BASE_DIR, pd_dir)
        if convert_to_paddlepaddle_simple(full_onnx_path, pd_output_dir, model_file):
            results['paddlepaddle'] += 1
        
        # 转换为 JAX
        jax_output_dir = os.path.join(BASE_DIR, jax_dir)
        if convert_to_jax_simple(full_onnx_path, jax_output_dir, model_file):
            results['jax'] += 1
    
    # 总结
    print("\n" + "="*70)
    print("Conversion Summary")
    print("="*70)
    print(f"MindSpore:   {results['mindspore']}/{len(MODELS)} models")
    print(f"PaddlePaddle: {results['paddlepaddle']}/{len(MODELS)} models")
    print(f"JAX:         {results['jax']}/{len(MODELS)} models")
    print("="*70)
    print("\nNote: These are wrapper models around ONNX Runtime.")
    print("For native framework models, use specialized conversion tools:")
    print("  - MindSpore: mindconverter")
    print("  - PaddlePaddle: x2paddle")
    print("  - JAX: onnx2jax (if available)")


if __name__ == "__main__":
    main()

