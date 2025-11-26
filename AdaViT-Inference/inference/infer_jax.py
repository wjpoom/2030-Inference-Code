import argparse
import numpy as np
import cv2
import os
import onnxruntime as ort
from PIL import Image

try:
    import jax
    import jax.numpy as jnp
except ImportError:
    print("JAX is not installed. Please install it to run this script.")
    exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='JAX Inference using ONNX Model')
    parser.add_argument('--model', default='inference/ada_deit_small.onnx', type=str, help='Path to ONNX model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    return parser.parse_args()

def preprocess(image_path):
    img = Image.open(image_path).convert('RGB')
    
    # Resize
    img = img.resize((256, 256), resample=Image.BICUBIC)
    
    # Center Crop
    left = (256 - 224) / 2
    top = (256 - 224) / 2
    right = (256 + 224) / 2
    bottom = (256 + 224) / 2
    img = img.crop((left, top, right, bottom))
    
    img_np = np.array(img).astype(np.float32) / 255.0
    
    # Normalize
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np - mean) / std
    
    # HWC to CHW
    img_np = img_np.transpose(2, 0, 1)
    
    # Add batch dimension
    img_np = np.expand_dims(img_np, axis=0)
    
    return jnp.array(img_np)

def main():
    args = parse_args()
    
    if not os.path.exists(args.model):
        print(f"Model file {args.model} not found. Please run convert_to_onnx.py first.")
        return

    print(f"Loading model from {args.model}...")
    session = ort.InferenceSession(args.model)
    
    input_name = session.get_inputs()[0].name
    
    print(f"Processing image {args.image}...")
    input_data = preprocess(args.image)
    
    # Convert JAX array to numpy for ONNX Runtime
    input_data_np = np.array(input_data)
    
    print("Running inference...")
    outputs = session.run(None, {input_name: input_data_np})
    
    logits = outputs[0]
    
    # Use JAX for softmax
    logits_jax = jnp.array(logits)
    probs = jax.nn.softmax(logits_jax, axis=1)
    
    pred_idx = jnp.argmax(probs, axis=1)[0]
    confidence = probs[0][pred_idx]
    
    print(f"Prediction: Class {pred_idx}, Confidence: {confidence:.4f}")

if __name__ == '__main__':
    main()

