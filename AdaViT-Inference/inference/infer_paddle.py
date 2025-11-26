import argparse
import numpy as np
import cv2
import os
import onnxruntime as ort
# Try importing paddle, handle if not present
try:
    import paddle
    from paddle.vision import transforms as T
except ImportError:
    print("PaddlePaddle is not installed. Please install it to run this script.")
    exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description='PaddlePaddle Inference using ONNX Model')
    parser.add_argument('--model', default='inference/ada_deit_small.onnx', type=str, help='Path to ONNX model')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    return parser.parse_args()

def preprocess(image_path):
    # Define transforms using Paddle
    val_transforms = T.Compose([
        T.Resize(256),
        T.CenterCrop(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not load image from {image_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Paddle transforms expect CHW or HWC? usually HWC for numpy before ToTensor
    # ToTensor converts to CHW and float32
    img_tensor = val_transforms(img)
    
    # Add batch dimension
    img_tensor = paddle.unsqueeze(img_tensor, axis=0)
    
    return img_tensor.numpy()

def main():
    args = parse_args()
    
    if not os.path.exists(args.model):
        print(f"Model file {args.model} not found. Please run convert_to_onnx.py first.")
        return

    # For true Paddle inference, one would typically convert ONNX to Paddle format using x2paddle
    # and then use paddle.jit.load or paddle.inference.
    # Here, we demonstrate using the ONNX model with Paddle preprocessing.
    
    print(f"Loading model from {args.model}...")
    session = ort.InferenceSession(args.model)
    
    input_name = session.get_inputs()[0].name
    
    print(f"Processing image {args.image}...")
    input_data = preprocess(args.image)
    
    print("Running inference...")
    outputs = session.run(None, {input_name: input_data})
    
    # Output index 0 is usually the class logits/probs
    # Based on convert_to_onnx.py: output_names = ["output", "head_select", "layer_select", "token_select", "select_logits"]
    logits = outputs[0]
    
    probs = paddle.nn.functional.softmax(paddle.to_tensor(logits), axis=1).numpy()
    pred_idx = np.argmax(probs, axis=1)[0]
    confidence = probs[0][pred_idx]
    
    print(f"Prediction: Class {pred_idx}, Confidence: {confidence:.4f}")

if __name__ == '__main__':
    main()

