# Inference Scripts for AdaViT

This directory contains scripts to convert the AdaViT PyTorch model to ONNX and run inference using PyTorch, PaddlePaddle, MindSpore, and JAX.

## Prerequisites

Ensure you have the necessary packages installed:

```bash
conda create -n adavit python=3.10 -y
conda activate adavit
pip install timm onnx onnxruntime opencv-python-headless onnxscript
# Note: You may need to adjust protobuf version if you encounter issues with ONNX
# pip install "protobuf<3.20"
```

For framework-specific scripts, you need the respective framework installed:
- PyTorch (required for conversion and PyTorch inference)
- PaddlePaddle
- MindSpore
- JAX

Download checkpoints to specified path:
| Ada-DeiT-S   |   77.3   | 2.3G  | [link](https://drive.google.com/file/d/1vkD6w9J8sf64IPhTBgyfvsTvUlZw6TNa/view?usp=sharing)|

Make sure the directory is like this:
```
AdaViT/
├── ckpt/
    └── ada_step_deit_small_patch16_224-224-adahlt.tar
```


## 1. PyTorch Inference

You can run inference directly using PyTorch to verify the model and checkpoint.

```bash
conda activate adavit

python inference/infer_torch.py --model ada_step_deit_small_patch16_224 --checkpoint ckpt/ada_step_deit_small_patch16_224-224-adahlt.tar --image inference/demo/dog1.png
```

## 2. Convert PyTorch Model to ONNX

Convert the PyTorch model checkpoint to ONNX format.

```bash
conda activate adavit

python inference/convert_to_onnx.py --model ada_step_deit_small_patch16_224 --checkpoint ckpt/ada_step_deit_small_patch16_224-224-adahlt.tar --output inference/ada_deit_small.onnx
```

**Note:** Ensure your checkpoint path is correct.

## 3. Run Inference with Other Frameworks

The following scripts use the exported ONNX model (`inference/ada_deit_small.onnx`).

### PaddlePaddle

```bash
conda activate adavit
# pip install paddlepaddle

python inference/infer_paddle.py --model inference/ada_deit_small.onnx --image inference/demo/dog1.png
```

### MindSpore

```bash
conda activate adavit
# pip install mindspore

python inference/infer_mindspore.py --model inference/ada_deit_small.onnx --image inference/demo/dog1.png
```

### JAX

```bash
conda activate adavit
# pip install jax

python inference/infer_jax.py --model inference/ada_deit_small.onnx --image inference/demo/dog1.png
```