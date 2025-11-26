import argparse
import torch
import sys
import os
from PIL import Image
from torchvision import transforms

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timm.models import create_model
import models 
from utils import ada_load_state_dict

def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Inference for AdaViT')
    parser.add_argument('--model', default='ada_step_deit_small_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to use')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='Path to checkpoint')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--img-size', type=int, default=224, metavar='N',
                        help='Image size')
    return parser.parse_args()

def preprocess_image(image_path, img_size):
    transform = transforms.Compose([
        transforms.Resize(int(img_size * 256 / 224), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)
    return img_tensor

def main():
    args = parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print(f"Creating model {args.model}...")
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        img_size=args.img_size,
        ada_layer=True,
        ada_head=True,
        ada_token=True,
    )
    
    model.to(device)
    model.eval()

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        ada_load_state_dict(args.checkpoint, model, use_qkv=False, strict=False)
    else:
        print("Warning: No checkpoint provided, using random weights.")

    img_tensor = preprocess_image(args.image, args.img_size).to(device)
    
    print(f"Processing image {args.image}...")
    
    with torch.no_grad():
        # Model forward returns a tuple: 
        # (x, head_select, layer_select, token_select, select_logits) or similar depending on config
        # We check the implementation of AdaStepT2T_ViT.forward
        outputs = model(img_tensor)
        
        # The first element is the class logits
        logits = outputs[0]
        
        probs = torch.nn.functional.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)
        
        print(f"Prediction: Class {pred_idx.item()}, Confidence: {confidence.item():.4f}")
        
        # Optional: Print selection stats if available
        if len(outputs) > 1:
             # outputs: x, head_select, layer_select, token_select, select_logtis
             if outputs[1] is not None:
                 print(f"Avg Head Usage: {outputs[1].mean().item():.2f}")
             if outputs[2] is not None:
                 print(f"Avg Layer Usage: {outputs[2].mean().item():.2f}")
             if outputs[3] is not None:
                 print(f"Avg Token Usage: {outputs[3].mean().item():.2f}")

if __name__ == '__main__':
    main()

