import argparse
import torch
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from timm.models import create_model
import models 
from utils import ada_load_state_dict

def parse_args():
    parser = argparse.ArgumentParser(description='Convert PyTorch model to ONNX')
    parser.add_argument('--model', default='ada_step_deit_small_patch16_224', type=str, metavar='MODEL',
                        help='Name of model to convert')
    parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                        help='Path to checkpoint')
    parser.add_argument('--output', default='model.onnx', type=str, metavar='PATH',
                        help='Path to output ONNX file')
    parser.add_argument('--img-size', type=int, default=224, metavar='N',
                        help='Image size')
    parser.add_argument('--batch-size', type=int, default=1, metavar='N',
                        help='Batch size')
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Creating model {args.model}...")
    # Initialize model with default parameters or those required by the checkpoint
    # You might need to adjust these parameters based on how the model was trained
    model = create_model(
        args.model,
        pretrained=False,
        num_classes=1000,
        img_size=args.img_size,
        # Add other necessary arguments here if needed, e.g., ada_head, ada_layer, etc.
        # These defaults should match the training config or the checkpoint
        ada_layer=True,
        ada_head=True,
        ada_token=True,
    )

    model.eval()

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        # Using the custom load function from utils.py as seen in ada_main.py
        ada_load_state_dict(args.checkpoint, model, use_qkv=False, strict=False)
    
    # Wrap model to handle tuple output and dictionary removal
    class ModelWrapper(torch.nn.Module):
        def __init__(self, model):
            super().__init__()
            self.model = model

        def forward(self, x):
            # Model returns: x, head_select, layer_select, token_select, select_logits (dict)
            outputs = self.model(x)
            # Return only the tensors, ignore the dictionary
            return outputs[0], outputs[1], outputs[2], outputs[3]

    model_wrapper = ModelWrapper(model)
    
    dummy_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size)
    
    print(f"Exporting to {args.output}...")
    
    input_names = ["input"]
    output_names = ["output", "head_select", "layer_select", "token_select"]
    
    try:
        torch.onnx.export(
            model_wrapper, 
            dummy_input, 
            args.output, 
            verbose=False, 
            input_names=input_names, 
            output_names=output_names,
            opset_version=17
        )
    except Exception as e:
        print(f"Export failed: {e}")
    
    print("Export complete.")

if __name__ == '__main__':
    main()

