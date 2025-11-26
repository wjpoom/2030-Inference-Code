import argparse
import numpy as np
import cv2
import os
import onnxruntime as ort

try:
    import mindspore
    import mindspore.dataset.vision as vision
    import mindspore.dataset.transforms as transforms
    from mindspore import Tensor
except ImportError:
    print("MindSpore is not installed. Please install it to run this script.")
    exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='MindSpore Inference using ONNX Model')
    parser.add_argument('--model', default='inference/ada_deit_small.onnx', type=str, help='Path to ONNX model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    return parser.parse_args()

def preprocess(image_path):
    # MindSpore vision operations
    mean_vec = [0.485 * 255, 0.456 * 255, 0.406 * 255]
    std_vec = [0.229 * 255, 0.224 * 255, 0.225 * 255]

    decode_op = vision.Decode()
    resize_op = vision.Resize(256)
    center_crop_op = vision.CenterCrop(224)
    normalize_op = vision.Normalize(mean=mean_vec, std=std_vec)
    hwc2chw_op = vision.HWC2CHW()
    
    img = np.fromfile(image_path, dtype=np.uint8)
    img = decode_op(img)
    img = resize_op(img)
    img = center_crop_op(img)
    img = normalize_op(img)
    img = hwc2chw_op(img)
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return img

def main():
    args = parse_args()
    
    if not os.path.exists(args.model):
        print(f"Model file {args.model} not found. Please run convert_to_onnx.py first.")
        return

    print(f"Loading model from {args.model}...")
    session = ort.InferenceSession(args.model)
    
    input_name = session.get_inputs()[0].name
    
    print(f"Processing image {args.image}...")
    # Ensure float32
    input_data = preprocess(args.image).astype(np.float32)
    
    print("Running inference...")
    outputs = session.run(None, {input_name: input_data})
    
    logits = outputs[0]
    
    # Use MindSpore for softmax to demonstrate usage
    logits_tensor = Tensor(logits)
    probs = mindspore.ops.Softmax(axis=1)(logits_tensor).asnumpy()
    
    pred_idx = np.argmax(probs, axis=1)[0]
    confidence = probs[0][pred_idx]
    
    print(f"Prediction: Class {pred_idx}, Confidence: {confidence:.4f}")

if __name__ == '__main__':
    main()

