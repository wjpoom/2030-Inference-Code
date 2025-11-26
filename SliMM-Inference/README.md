# SliMM-Inference: Multi-Framework Inference for SliMM Models

本项目提供了将 PyTorch 版本的 SliMM 模型转换为 ONNX 格式，并进一步转换为 MindSpore、PaddlePaddle 和 JAX 三种框架的能力，同时支持在四种框架（PyTorch、ONNX Runtime、MindSpore、PaddlePaddle、JAX）上进行推理。

## 🌟 核心功能

- **PyTorch → ONNX**: 将 PyTorch 模型导出为 ONNX 格式
- **ONNX → 多框架**: 支持将 ONNX 模型转换为 MindSpore、PaddlePaddle、JAX 三种框架
- **多框架推理**: 支持在 PyTorch、ONNX Runtime、MindSpore、PaddlePaddle、JAX 五种框架上进行推理
- **ONNX Runtime 包装器**: 提供统一的包装器接口，简化跨框架使用

## 📋 目录结构

```
SliMM-Inference/
├── export_slimm.py              # PyTorch 模型导出为 ONNX
├── convert_onnx_to_frameworks.py # ONNX 转换为其他框架
├── convert_models.py            # 模型转换工具
├── onnx_wrappers.py             # ONNX Runtime 包装器
├── inference_torch.py           # PyTorch 推理脚本
├── inference_onnx.py            # ONNX Runtime 推理脚本
├── inference_mindspore.py       # MindSpore 推理脚本
├── inference_paddlepaddle.py    # PaddlePaddle 推理脚本
├── inference_jax.py             # JAX 推理脚本
├── CONVERSION_README.md         # 转换详细说明
└── slimm/                       # SliMM 模型代码
```

## 🚀 快速开始

### 1. 安装依赖

```bash
# 创建 conda 环境
conda create -n slimm python=3.10 -y
conda activate slimm

# 安装基础依赖
pip install --upgrade pip
pip install -e .

# 安装 transformers（需要特定版本）
pip install transformers@git+https://github.com/huggingface/transformers.git@7bbc62474391aff64f63fcc064c975752d1fa4de

# 安装 ONNX Runtime（必需）
pip install onnxruntime

# 可选：安装其他框架（根据需要选择）
# MindSpore
pip install mindspore

# PaddlePaddle
pip install paddlepaddle

# JAX
pip install jax jaxlib
```

### 2. 模型转换流程

#### 步骤 1: PyTorch → ONNX

首先将 PyTorch 模型导出为 ONNX 格式：

```bash
# 编辑 export_slimm.py，设置模型路径
# path = '/path/to/SliMM-Qwen2-0.5B'
# onnx_model_A = '/path/to/slimm_onnx/SliMM_A.onnx'
# ... 设置其他模型路径

python export_slimm.py
```

这将生成以下 ONNX 模型文件：
- `SliMM_A.onnx` - 文本编码器
- `SliMM_B.onnx` - 视觉编码器
- `SliMM_C.onnx` - 多模态融合
- `SliMM_D.onnx` - Rotary 位置编码（有视觉）
- `SliMM_E.onnx` - Rotary 位置编码（无视觉）
- `SliMM_F.onnx` - LLM 解码器

#### 步骤 2: ONNX → 其他框架

使用 `convert_onnx_to_frameworks.py` 脚本进行转换：

```bash
# 编辑脚本中的路径配置
python convert_onnx_to_frameworks.py
```

详细说明请参考 [CONVERSION_README.md](CONVERSION_README.md)。

### 3. 运行推理

转换完成后，可以使用相应的推理脚本进行推理：

#### PyTorch 推理

```bash
python inference_torch.py
```

#### ONNX Runtime 推理

```bash
python inference_onnx.py
```

#### MindSpore 推理

```bash
python inference_mindspore.py
```

#### PaddlePaddle 推理

```bash
python inference_paddlepaddle.py
```

#### JAX 推理

```bash
python inference_jax.py
```

## 📝 使用示例

### PyTorch 推理示例

```python
from slimm.model.processor import SliMMQwen2VLProcessor
from slimm.model.slimm import SliMMForConditionalGeneration
from slimm.model.utils_vl import process_vision_info

model_path = "ckpt/SliMM-DeepStackE-Qwen2VL-2B"

model = SliMMForConditionalGeneration.from_pretrained(
    model_path, torch_dtype="auto", device_map="cuda:0"
)

processor = SliMMQwen2VLProcessor.from_pretrained(model_path)

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "demo.jpeg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# 准备输入
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# 推理
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_text)
```
### 推理参数配置

```python
INPUT_IMAGE_SIZE = [960, 960]      # 输入图像尺寸
HEIGHT_FACTOR = 15                 # 高度因子
WIDTH_FACTOR = 15                  # 宽度因子
MAX_SEQ_LENGTH = 4096              # 最大序列长度
STOP_TOKEN = [151643, 151645]      # 停止 token
```

## 🛠️ 技术细节

### ONNX Runtime 包装器

包装器类提供了统一的接口，使得在不同框架中使用 ONNX 模型变得简单：

- **MindSporeWrapper**: 将 ONNX 输出转换为 MindSpore Tensor
- **PaddlePaddleWrapper**: 将 ONNX 输出转换为 PaddlePaddle Tensor
- **JAXWrapper**: 将 ONNX 输出转换为 JAX Array

所有包装器都支持：
- 自动类型转换
- GPU/CPU 自动选择
- 输入输出名称自动获取
- 序列化支持（可保存为 .pkl 文件）

### 模型分割策略

SliMM 模型被分割为 6 个子模型：

1. **SliMM_A**: 文本编码器（Embedding + 部分 Transformer）
2. **SliMM_B**: 视觉编码器（Vision Encoder）
3. **SliMM_C**: 多模态融合层
4. **SliMM_D**: Rotary 位置编码（有视觉输入时使用）
5. **SliMM_E**: Rotary 位置编码（无视觉输入时使用）
6. **SliMM_F**: LLM 解码器（主要 Transformer 层）

这种分割策略使得：
- 可以灵活组合不同的子模型
- 支持增量推理（KV Cache）
- 便于在不同框架中实现

## ⚠️ 注意事项

1. **依赖版本**: 确保使用正确版本的 transformers（commit 7bbc6247）
2. **模型路径**: 确保所有模型路径配置正确
3. **框架兼容性**: 不同框架的 tensor 类型可能不同，包装器会自动处理
4. **GPU 支持**: 如需使用 GPU，确保安装了相应的 CUDA 版本和框架 GPU 版本
5. **内存管理**: 大模型推理时注意内存使用，必要时使用 CPU 推理

## 🐛 故障排除

### 转换失败

- 检查 ONNX 模型文件是否存在
- 确保已安装所需的依赖包
- 查看错误信息，可能需要调整输入/输出节点名称

### 推理失败

- 确保已创建包装器文件（.pkl）
- 检查模型路径是否正确
- 确保已安装相应框架
- 检查输入/输出形状是否匹配
- 查看框架特定的错误信息

### 性能问题

- ONNX Runtime 包装器性能可能略低于原生框架
- 如需最佳性能，建议使用原生框架转换
- GPU 推理通常比 CPU 快很多

## 📚 相关资源

- [SliMM 项目主页](https://deepstack-vl.github.io/)
- [SliMM 论文](https://arxiv.org/abs/2406.04334)
- [HuggingFace 模型](https://huggingface.co/collections/menglc/slimm-675bd737c2965037a6b52d05)
- [ONNX Runtime 文档](https://onnxruntime.ai/)
- [MindSpore 文档](https://www.mindspore.cn/)
- [PaddlePaddle 文档](https://www.paddlepaddle.org.cn/)
- [JAX 文档](https://jax.readthedocs.io/)

## 📄 许可证

请查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

本项目基于以下优秀项目：
- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL)
- [LLaVA](https://github.com/haotian-liu/LLaVA)
- [LLaVA-NeXT](https://github.com/LLaVA-VL/LLaVA-NeXT)

## 📧 联系方式

如有问题或建议，请提交 Issue 或 Pull Request。

---

**注意**: 本项目专注于推理功能，训练相关代码请参考主 SliMM 项目。
