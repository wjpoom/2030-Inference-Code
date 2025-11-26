# 模型转换说明

本目录包含将 ONNX 模型转换为 MindSpore、PaddlePaddle 和 JAX 格式的脚本。

## 快速开始

### 方法 1: 使用 ONNX Runtime 包装器（推荐，最简单）

这种方法创建包装器，在相应框架中使用 ONNX Runtime 进行推理：

```bash
# 确保已安装 onnxruntime
pip install onnxruntime

# 运行包装器创建脚本
python create_wrappers.py
```

这将创建以下目录和文件：
- `slimm_mindspore/` - MindSpore 包装器 (.pkl)
- `slimm_paddle/` - PaddlePaddle 包装器 (.pkl)
- `slimm_jax/` - JAX 包装器 (.pkl)

### 方法 2: 使用原生框架转换工具

#### MindSpore

```bash
# 安装 MindConverter
pip install mindinsight
pip install onnx>=1.8.0 onnxoptimizer>=0.1.2 onnxruntime>=1.5.2

# 转换单个模型
mindconverter --model_file slimm_onnx/SliMM_A.onnx \
              --shape 1,512 \
              --input_nodes input_ids \
              --output_nodes output \
              --output slimm_mindspore/

# 注意：mindconverter 会生成 Python 代码，需要手动编译为 .mindir 格式
```

#### PaddlePaddle

```bash
# 安装 x2paddle
pip install x2paddle

# 转换模型
x2paddle --framework=onnx --model=slimm_onnx/SliMM_A.onnx --save_dir=slimm_paddle/SliMM_A

# 然后使用 paddle.jit.save() 创建推理模型
```

#### JAX

JAX 没有官方的 ONNX 转换工具。可以使用：
1. 使用包装器方法（方法 1）
2. 手动重写模型为 JAX/Flax 代码
3. 使用第三方工具如 `onnx2jax`（如果可用）

## 文件说明

- `create_wrappers.py` - 创建 ONNX Runtime 包装器（最简单的方法）
- `convert_models.py` - 完整的转换脚本（需要安装相应框架）
- `convert_onnx_to_frameworks.py` - 详细的转换脚本

## 使用转换后的模型

转换完成后，推理代码会自动检测并使用：
1. 首先尝试加载包装器文件 (.pkl)
2. 如果不存在，尝试加载原生框架模型

### 运行推理

```bash
# MindSpore
python inference_mindspore.py

# PaddlePaddle
python inference_paddlepaddle.py

# JAX
python inference_jax.py
```

## 注意事项

1. **包装器方法**：使用 ONNX Runtime 作为后端，性能可能略低于原生框架，但兼容性最好
2. **原生转换**：需要安装相应的框架和转换工具，转换过程可能较复杂
3. **模型路径**：确保 ONNX 模型文件存在于 `slimm_onnx/` 和 `slimm_onnx_2/` 目录中

## 故障排除

### 如果转换失败

1. 检查 ONNX 模型文件是否存在
2. 确保已安装所需的依赖包
3. 查看错误信息，可能需要调整输入/输出节点名称
4. 对于 MindSpore，可能需要手动处理不支持的算子

### 如果推理失败

1. 确保已运行 `create_wrappers.py` 创建包装器
2. 检查模型路径是否正确
3. 确保已安装相应框架（MindSpore/PaddlePaddle/JAX）
4. 检查输入/输出形状是否匹配

## 模型文件结构

转换后的目录结构：

```
SliMM/
├── slimm_onnx/          # 原始 ONNX 模型
│   ├── SliMM_A.onnx
│   ├── SliMM_B.onnx
│   ├── SliMM_C.onnx
│   ├── SliMM_D.onnx
│   └── SliMM_E.onnx
├── slimm_onnx_2/
│   └── SliMM_F.onnx
├── slimm_mindspore/     # MindSpore 模型/包装器
│   ├── SliMM_A.pkl
│   └── ...
├── slimm_paddle/        # PaddlePaddle 模型/包装器
│   └── ...
└── slimm_jax/           # JAX 模型/包装器
    └── ...
```

