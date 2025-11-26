#!/usr/bin/env python3
"""
将 ONNX 模型转换为 MindSpore、PaddlePaddle 和 JAX 格式的脚本
"""

import os
import sys
import subprocess
import onnxruntime as ort
import numpy as np
from pathlib import Path

# 模型路径配置
BASE_DIR = "/mnt/public/users/chenyitong/projects/SliMM"
ONNX_MODELS = {
    'A': f"{BASE_DIR}/slimm_onnx/SliMM_A.onnx",
    'B': f"{BASE_DIR}/slimm_onnx/SliMM_B.onnx",
    'C': f"{BASE_DIR}/slimm_onnx/SliMM_C.onnx",
    'D': f"{BASE_DIR}/slimm_onnx/SliMM_D.onnx",
    'E': f"{BASE_DIR}/slimm_onnx/SliMM_E.onnx",
    'F': f"{BASE_DIR}/slimm_onnx_2/SliMM_F.onnx",
}

OUTPUT_DIRS = {
    'mindspore': f"{BASE_DIR}/slimm_mindspore",
    'mindspore_2': f"{BASE_DIR}/slimm_mindspore_2",
    'paddle': f"{BASE_DIR}/slimm_paddle",
    'paddle_2': f"{BASE_DIR}/slimm_paddle_2",
    'jax': f"{BASE_DIR}/slimm_jax",
    'jax_2': f"{BASE_DIR}/slimm_jax_2",
}


def get_onnx_input_output_info(onnx_path):
    """获取 ONNX 模型的输入输出信息"""
    try:
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        input_info = {}
        for inp in inputs:
            input_info[inp.name] = {
                'shape': inp.shape,
                'type': inp.type
            }
        
        output_info = {}
        for out in outputs:
            output_info[out.name] = {
                'shape': out.shape,
                'type': out.type
            }
        
        return input_info, output_info
    except Exception as e:
        print(f"Error reading ONNX model {onnx_path}: {e}")
        return {}, {}


def convert_to_mindspore(onnx_path, output_path, model_name):
    """将 ONNX 模型转换为 MindSpore 格式"""
    print(f"\n{'='*60}")
    print(f"Converting {model_name} to MindSpore format...")
    print(f"{'='*60}")
    
    # 获取模型输入输出信息
    input_info, output_info = get_onnx_input_output_info(onnx_path)
    
    if not input_info:
        print(f"Warning: Could not read ONNX model {onnx_path}, skipping...")
        return False
    
    # 获取第一个输入的形状（用于 mindconverter）
    first_input = list(input_info.keys())[0]
    input_shape = input_info[first_input]['shape']
    
    # 处理动态维度（用 -1 表示）
    shape_str = ','.join([str(dim) if dim is not None and dim > 0 else '1' for dim in input_shape])
    
    # 使用 mindconverter 转换
    # 注意：mindconverter 会生成 Python 代码，需要手动编译为 .mindir
    try:
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        cmd = [
            'mindconverter',
            '--model_file', onnx_path,
            '--shape', shape_str,
            '--input_nodes', first_input,
            '--output_nodes', ','.join(output_info.keys()),
            '--output', output_dir
        ]
        
        print(f"Running command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✓ MindSpore conversion successful for {model_name}")
            print(f"  Output directory: {output_dir}")
            print(f"  Note: You may need to compile the generated Python code to .mindir format")
            return True
        else:
            print(f"✗ MindSpore conversion failed for {model_name}")
            print(f"  Error: {result.stderr}")
            return False
            
    except FileNotFoundError:
        print(f"✗ mindconverter not found. Please install it with: pip install mindinsight")
        print(f"  Then install dependencies: pip install onnx>=1.8.0 onnxoptimizer>=0.1.2 onnxruntime>=1.5.2")
        return False
    except Exception as e:
        print(f"✗ Error during MindSpore conversion: {e}")
        return False


def convert_to_paddlepaddle(onnx_path, output_path, model_name):
    """将 ONNX 模型转换为 PaddlePaddle 格式"""
    print(f"\n{'='*60}")
    print(f"Converting {model_name} to PaddlePaddle format...")
    print(f"{'='*60}")
    
    try:
        import paddle
        from x2paddle.convert import onnx2paddle
        
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        # 使用 x2paddle 转换
        onnx2paddle(onnx_path, output_dir, convert_to_lite=False, lite_valid_places="arm")
        
        print(f"✓ PaddlePaddle conversion successful for {model_name}")
        print(f"  Output directory: {output_dir}")
        print(f"  Note: You may need to use paddle.jit.save() to create inference model")
        return True
        
    except ImportError:
        print(f"✗ x2paddle not found. Please install it with: pip install x2paddle")
        return False
    except Exception as e:
        print(f"✗ Error during PaddlePaddle conversion: {e}")
        # 尝试使用 PaddlePaddle 的 ONNX 导入功能
        try:
            import paddle
            from paddle import fluid
            
            # 另一种方法：使用 PaddlePaddle 的 ONNX 导入
            print(f"  Trying alternative method using paddle.onnx...")
            # 注意：PaddlePaddle 可能需要先加载 ONNX 模型然后保存为 Paddle 格式
            return False
        except:
            return False


def convert_to_jax(onnx_path, output_path, model_name):
    """将 ONNX 模型转换为 JAX 格式"""
    print(f"\n{'='*60}")
    print(f"Converting {model_name} to JAX format...")
    print(f"{'='*60}")
    
    try:
        import onnx
        import jax
        import jax.numpy as jnp
        import pickle
        
        # 加载 ONNX 模型
        onnx_model = onnx.load(onnx_path)
        
        # 获取模型信息
        input_info, output_info = get_onnx_input_output_info(onnx_path)
        
        # 创建一个简单的 JAX 包装器
        # 注意：这是一个简化的方法，实际转换可能需要更复杂的处理
        def create_jax_wrapper(onnx_path):
            """创建一个使用 ONNX Runtime 的 JAX 包装器"""
            session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
            
            def jax_forward(*args):
                # 将 JAX 数组转换为 numpy
                inputs = {}
                input_names = [inp.name for inp in session.get_inputs()]
                for i, arg in enumerate(args):
                    if i < len(input_names):
                        inputs[input_names[i]] = np.array(arg)
                
                # 运行 ONNX Runtime
                outputs = session.run(None, inputs)
                
                # 转换回 JAX 数组
                if len(outputs) == 1:
                    return jnp.array(outputs[0])
                else:
                    return [jnp.array(out) for out in outputs]
            
            return jax_forward
        
        # 创建包装器并保存
        jax_model = create_jax_wrapper(onnx_path)
        
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = output_path.replace('.mindir', '.pkl').replace('.pdmodel', '.pkl')
        with open(output_file, 'wb') as f:
            pickle.dump(jax_model, f)
        
        print(f"✓ JAX conversion successful for {model_name}")
        print(f"  Output file: {output_file}")
        print(f"  Note: This is a wrapper around ONNX Runtime. For native JAX, consider using onnx2jax")
        return True
        
    except ImportError as e:
        print(f"✗ Required packages not found: {e}")
        print(f"  Please install: pip install onnx jax jaxlib")
        return False
    except Exception as e:
        print(f"✗ Error during JAX conversion: {e}")
        return False


def main():
    """主函数"""
    print("="*60)
    print("ONNX to Multiple Frameworks Converter")
    print("="*60)
    
    # 创建输出目录
    for dir_path in OUTPUT_DIRS.values():
        os.makedirs(dir_path, exist_ok=True)
    
    # 检查 ONNX 模型是否存在
    missing_models = []
    for name, path in ONNX_MODELS.items():
        if not os.path.exists(path):
            missing_models.append((name, path))
    
    if missing_models:
        print("\nWarning: The following ONNX models are missing:")
        for name, path in missing_models:
            print(f"  {name}: {path}")
        print("\nPlease ensure all ONNX models exist before conversion.")
        return
    
    # 转换每个模型
    results = {
        'mindspore': {'success': 0, 'failed': 0},
        'paddlepaddle': {'success': 0, 'failed': 0},
        'jax': {'success': 0, 'failed': 0},
    }
    
    for model_name, onnx_path in ONNX_MODELS.items():
        print(f"\n\nProcessing model: {model_name}")
        print(f"ONNX path: {onnx_path}")
        
        # 确定输出路径
        if model_name == 'F':
            mindspore_output = f"{OUTPUT_DIRS['mindspore_2']}/SliMM_{model_name}.mindir"
            paddle_output = f"{OUTPUT_DIRS['paddle_2']}/SliMM_{model_name}"
            jax_output = f"{OUTPUT_DIRS['jax_2']}/SliMM_{model_name}.pkl"
        else:
            mindspore_output = f"{OUTPUT_DIRS['mindspore']}/SliMM_{model_name}.mindir"
            paddle_output = f"{OUTPUT_DIRS['paddle']}/SliMM_{model_name}"
            jax_output = f"{OUTPUT_DIRS['jax']}/SliMM_{model_name}.pkl"
        
        # 转换为 MindSpore
        if convert_to_mindspore(onnx_path, mindspore_output, f"SliMM_{model_name}"):
            results['mindspore']['success'] += 1
        else:
            results['mindspore']['failed'] += 1
        
        # 转换为 PaddlePaddle
        if convert_to_paddlepaddle(onnx_path, paddle_output, f"SliMM_{model_name}"):
            results['paddlepaddle']['success'] += 1
        else:
            results['paddlepaddle']['failed'] += 1
        
        # 转换为 JAX
        if convert_to_jax(onnx_path, jax_output, f"SliMM_{model_name}"):
            results['jax']['success'] += 1
        else:
            results['jax']['failed'] += 1
    
    # 打印总结
    print("\n\n" + "="*60)
    print("Conversion Summary")
    print("="*60)
    for framework, stats in results.items():
        print(f"{framework.upper()}:")
        print(f"  Success: {stats['success']}/{len(ONNX_MODELS)}")
        print(f"  Failed: {stats['failed']}/{len(ONNX_MODELS)}")
    print("="*60)


if __name__ == "__main__":
    main()

