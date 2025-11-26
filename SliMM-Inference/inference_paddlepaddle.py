from transformers import AutoTokenizer
import paddle
import numpy as np
import os
from PIL import Image
import time
import pickle
from onnx_wrappers import PaddlePaddleWrapper  # 导入包装器类

def is_valid_image_path(image_path):
    if not os.path.exists(image_path):
        return False
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    _, ext = os.path.splitext(image_path)
    return ext.lower() in valid_extensions

# Helper function to create PaddlePaddle tensor with version compatibility
def to_tensor(data, dtype=None):
    """创建 PaddlePaddle tensor，兼容不同版本"""
    # 确保数据是 numpy array
    if not isinstance(data, np.ndarray):
        data = np.array(data)
    
    # 处理 dtype
    if dtype:
        # 将 dtype 字符串转换为 numpy dtype
        dtype_map = {
            'int32': np.int32,
            'int64': np.int64,
            'int8': np.int8,
            'uint8': np.uint8,
            'float32': np.float32,
            'float64': np.float64,
        }
        np_dtype = dtype_map.get(dtype, dtype)
        if data.dtype != np_dtype:
            data = data.astype(np_dtype)
    
    # 尝试使用 paddle.to_tensor (PaddlePaddle 2.0+)
    if hasattr(paddle, 'to_tensor'):
        try:
            if dtype:
                return paddle.to_tensor(data, dtype=dtype)
            else:
                return paddle.to_tensor(data)
        except:
            pass
    
    # 回退方案：使用 paddle.Tensor 或 paddle.cast
    try:
        tensor = paddle.Tensor(data)
        if dtype and hasattr(paddle, 'cast'):
            return paddle.cast(tensor, dtype=dtype)
        return tensor
    except:
        # 最后的回退：直接使用 numpy array（如果模型接受）
        return data

# Helper function to convert tensor to numpy safely
def to_numpy(tensor_or_array):
    """安全地将 PaddlePaddle tensor 或 numpy array 转换为 numpy array"""
    if isinstance(tensor_or_array, np.ndarray):
        return tensor_or_array
    elif hasattr(tensor_or_array, 'numpy'):
        return tensor_or_array.numpy()
    else:
        return np.array(tensor_or_array)

# PaddlePaddle settings
# paddle.set_device('cpu')

# Run the exported model by PaddlePaddle
path = r'/mnt/public/users/chenyitong/projects/SliMM/ckpt/SliMM-Qwen2-0.5B-Tokenzier'
# Model paths - supports both .pkl (wrapper) and native PaddlePaddle formats
paddle_model_A = r'/mnt/public/users/chenyitong/projects/SliMM/slimm_paddle/SliMM_A.pkl'    # ONNX Runtime wrapper (created by create_wrappers.py)
paddle_model_B = r'/mnt/public/users/chenyitong/projects/SliMM/slimm_paddle/SliMM_B.pkl'
paddle_model_C = r'/mnt/public/users/chenyitong/projects/SliMM/slimm_paddle/SliMM_C.pkl'
paddle_model_D = r'/mnt/public/users/chenyitong/projects/SliMM/slimm_paddle/SliMM_D.pkl'
paddle_model_E = r'/mnt/public/users/chenyitong/projects/SliMM/slimm_paddle/SliMM_E.pkl'
paddle_model_F = r'/mnt/public/users/chenyitong/projects/SliMM/slimm_paddle_2/SliMM_F.pkl'  # ONNX Runtime wrapper
max_single_chat_length = 4096                         # It an adjustable value, but must less than max_seq_len.
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)


# Load PaddlePaddle models
# Option 1: Load native PaddlePaddle models
# Option 2: Load ONNX Runtime wrappers (.pkl format)
import pickle

def load_model(model_path):
    """加载模型，支持 PaddlePaddle 原生格式和 .pkl (wrapper) 格式"""
    if model_path.endswith('.pkl'):
        # 加载 ONNX Runtime 包装器
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        # 加载原生 PaddlePaddle 模型
        return paddle.jit.load(model_path)

try:
    model_A = load_model(paddle_model_A)
    model_B = load_model(paddle_model_B)
    model_C = load_model(paddle_model_C)
    model_D = load_model(paddle_model_D)
    model_E = load_model(paddle_model_E)
    model_F = load_model(paddle_model_F)
except FileNotFoundError:
    # 如果包装器不存在，尝试加载原生模型
    try:
        model_A = paddle.jit.load(paddle_model_A)
        model_B = paddle.jit.load(paddle_model_B)
        model_C = paddle.jit.load(paddle_model_C)
        model_D = paddle.jit.load(paddle_model_D)
        model_E = paddle.jit.load(paddle_model_E)
        model_F = paddle.jit.load(paddle_model_F)
    except:
        raise ValueError("Please run create_wrappers.py first to create model wrappers, or provide native PaddlePaddle models")

# Get input/output names from actual model objects
# PaddlePaddleWrapper models have output_names attribute
in_name_A0 = "input_ids"
# Get actual output name from model_A
if hasattr(model_A, 'output_names') and len(model_A.output_names) > 0:
    out_name_A0 = model_A.output_names[0]
else:
    out_name_A0 = "output"  # Fallback

in_name_B0 = "pixel_values"
# Get actual output name from model_B
if hasattr(model_B, 'output_names') and len(model_B.output_names) > 0:
    out_name_B0 = model_B.output_names[0]
else:
    out_name_B0 = "output"  # Fallback

in_name_C0 = "hidden_states"
in_name_C1 = "vision_hidden_states"
# Get actual output name from model_C
if hasattr(model_C, 'output_names') and len(model_C.output_names) > 0:
    out_name_C0 = model_C.output_names[0]
else:
    out_name_C0 = "output"  # Fallback

in_name_D0 = "history_len"
in_name_D1 = "kv_seq_len"
# Get actual output names from model_D
if hasattr(model_D, 'output_names') and len(model_D.output_names) > 0:
    out_name_D = model_D.output_names
    rotary_outputs_len = len(out_name_D)
else:
    rotary_outputs_len = 2  # Fallback
    out_name_D = [f"output_{i}" for i in range(rotary_outputs_len)]

in_name_E0 = "history_len"
in_name_E1 = "kv_seq_len"
# Get actual output names from model_E
if hasattr(model_E, 'output_names') and len(model_E.output_names) > 0:
    out_name_E = model_E.output_names
    # Ensure rotary_outputs_len matches
    if len(out_name_E) != rotary_outputs_len:
        rotary_outputs_len = len(out_name_E)
else:
    out_name_E = [f"output_{i}" for i in range(rotary_outputs_len)]

# For model F, we need to determine the number of layers from model structure
# Get actual output names from model_F
if hasattr(model_F, 'output_names') and len(model_F.output_names) > 0:
    out_name_F = model_F.output_names
    amount_of_outputs_F = len(out_name_F)
else:
    # Fallback: use placeholder
    amount_of_outputs_F = 50
    out_name_F = [f"output_{i}" for i in range(amount_of_outputs_F + 1)]

num_layers = (amount_of_outputs_F - 2) // 2
num_keys_values = num_layers + num_layers
num_keys_values_plus = num_keys_values + 1
rotary_indices = np.arange(num_keys_values, num_keys_values + rotary_outputs_len, dtype=np.int32) + 2
amount_of_outputs_F -= 1

# Get actual input names from model_F if available
if hasattr(model_F, 'input_names') and len(model_F.input_names) > 0:
    in_name_F = model_F.input_names
else:
    # Fallback: generate placeholder names
    in_name_F = [f"input_{i}" for i in range(num_keys_values + rotary_outputs_len + 3)]

image_path = r"demo.jpeg"
query = "Describe this image."
INPUT_IMAGE_SIZE = [960, 960]                                       # Input image shape. Should be a multiple of GPU group (e.g., 16) for optimal efficiency.
HEIGHT_FACTOR = 15                                                  # Adjust this value to determine the resize shape and vision resolution.
WIDTH_FACTOR = 15                                                   # Adjust this value to determine the resize shape and vision resolution.
MAX_SEQ_LENGTH = 4096                                               # The max token length. Note, this value include the 10 tokens for system prompt and (HEIGHT_FACTOR * WIDTH_FACTOR) tokens for image prompt. Hence, only (MAX_SEQ_LENGTH - (HEIGHT_FACTOR * WIDTH_FACTOR) - 10) tokens for query + response.
IMAGE_RESIZE = [HEIGHT_FACTOR * 28, WIDTH_FACTOR * 28]              # 28 = self.patch_size * self.merge_size
STOP_TOKEN = [151643, 151645]

num_decode = 0
prompt = f"<|im_start|>user\n<|vision_start|><|vision_end|>{query}<|im_end|>\n<|im_start|>assistant\n"
prompt_head_len = np.array([4], dtype=np.int64)
image_embed_size = WIDTH_FACTOR * HEIGHT_FACTOR
tokens = tokenizer(prompt, return_tensors='np')['input_ids'].astype(np.int32)
input_ids = to_tensor(tokens, dtype='int32')
ids_len = to_tensor(np.array([tokens.shape[-1]], dtype=np.int64), dtype='int64')
history_len = to_tensor(np.array([0], dtype=np.int64), dtype='int64')
kv_seq_len = to_tensor(to_numpy(ids_len) + to_numpy(history_len), dtype='int64')
attention_mask = to_tensor(np.array([1], dtype=np.int8), dtype='int8')
max_single_chat_length -= tokens.shape[-1]

# Initialize past keys and values
# Get shapes from model F's input metadata (similar to ONNX version)
try:
    # Get model_F's session to access input metadata
    session_F = model_F.session
    inputs_meta = session_F.get_inputs()
    
    # Get shapes from first key input and first value input
    # past_keys_F shape: (inputs_meta[0].shape[0], 1, inputs_meta[0].shape[2], 0)
    # past_values_F shape: (inputs_meta[num_layers].shape[0], 1, 0, inputs_meta[num_layers].shape[3])
    key_input_shape = inputs_meta[0].shape
    value_input_shape = inputs_meta[num_layers].shape
    
    # Handle dynamic dimensions (None or negative)
    def get_shape_dim(shape, idx, default):
        if idx < len(shape):
            dim = shape[idx]
            if dim is None or dim < 0:
                return default
            return dim
        return default
    
    # past_keys_F: (batch, 1, num_heads, 0)
    past_keys_shape = (
        get_shape_dim(key_input_shape, 0, 2),  # batch
        1,
        get_shape_dim(key_input_shape, 2, 32),  # num_heads
        0  # seq_len (initial 0)
    )
    
    # past_values_F: (batch, 1, 0, head_dim)
    past_values_shape = (
        get_shape_dim(value_input_shape, 0, 2),  # batch
        1,
        0,  # seq_len (initial 0)
        get_shape_dim(value_input_shape, 3, 64)  # head_dim
    )
    
    # Create single past_key and past_value (all layers share the same shape)
    # Note: Each layer gets its own input, so we don't add num_layers dimension
    past_keys_F = to_tensor(np.zeros(past_keys_shape, dtype=np.float32), dtype='float32')
    past_values_F = to_tensor(np.zeros(past_values_shape, dtype=np.float32), dtype='float32')
    
    print(f"Initialized past_keys_F shape: {past_keys_shape}")
    print(f"Initialized past_values_F shape: {past_values_shape}")
    
except Exception as e:
    print(f"Warning: Could not infer shapes from model_F: {e}")
    print("Using default shapes - this may cause errors")
    # Default shapes (based on error message: batch=2, num_heads=32, head_dim=64)
    past_keys_shape = (2, 1, 32, 0)  # (batch=2, 1, num_heads=32, seq_len=0)
    past_values_shape = (2, 1, 0, 64)  # (batch=2, 1, seq_len=0, head_dim=64)
    past_keys_F = to_tensor(np.zeros(past_keys_shape, dtype=np.float32), dtype='float32')
    past_values_F = to_tensor(np.zeros(past_values_shape, dtype=np.float32), dtype='float32')


# Load input image
if is_valid_image_path(image_path):
    image = Image.open(image_path)
    image = image.resize((INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    pixel_values = np.transpose(np.array(image).astype(np.uint8), (2, 0, 1))
    pixel_values = np.expand_dims(pixel_values, axis=0)
    use_vision = True
    print('\nChat with image.')
else:
    use_vision = False
    print('\nChat without image.')


# Start to run LLM
# Note: Adjust model calling based on your actual PaddlePaddle model interface
hidden_states = model_A(input_ids)
if isinstance(hidden_states, dict):
    if out_name_A0 in hidden_states:
        hidden_states = hidden_states[out_name_A0]
    else:
        # If expected key not found, try to use the first available key
        available_keys = list(hidden_states.keys())
        if available_keys:
            print(f"Warning: Expected output key '{out_name_A0}' not found. Using '{available_keys[0]}' instead.")
            print(f"Available keys: {available_keys}")
            hidden_states = hidden_states[available_keys[0]]
        else:
            raise KeyError(f"Model A returned empty dict. Expected key: '{out_name_A0}'")


if use_vision:
    ids_len = to_tensor(to_numpy(ids_len) + image_embed_size, dtype='int64')
    kv_seq_len = to_tensor(to_numpy(kv_seq_len) + image_embed_size, dtype='int64')

    max_single_chat_length -= image_embed_size

    print('\nStart to Process the Image...')
    start_time = time.time()
    pixel_values_tensor = to_tensor(pixel_values, dtype='uint8')
    vision_hidden_states = model_B(pixel_values_tensor)
    if isinstance(vision_hidden_states, dict):
        if out_name_B0 in vision_hidden_states:
            vision_hidden_states = vision_hidden_states[out_name_B0]
        else:
            available_keys = list(vision_hidden_states.keys())
            if available_keys:
                print(f"Warning: Expected output key '{out_name_B0}' not found in model_B. Using '{available_keys[0]}' instead.")
                vision_hidden_states = vision_hidden_states[available_keys[0]]
            else:
                raise KeyError(f"Model B returned empty dict. Expected key: '{out_name_B0}'")
    print(f'\nImage Process Complete. Time Cost: {time.time() - start_time:.3f} Seconds')

    hidden_states = model_C(hidden_states, vision_hidden_states)
    if isinstance(hidden_states, dict):
        if out_name_C0 in hidden_states:
            hidden_states = hidden_states[out_name_C0]
        else:
            available_keys = list(hidden_states.keys())
            if available_keys:
                print(f"Warning: Expected output key '{out_name_C0}' not found in model_C. Using '{available_keys[0]}' instead.")
                hidden_states = hidden_states[available_keys[0]]
            else:
                raise KeyError(f"Model C returned empty dict. Expected key: '{out_name_C0}'")

    rotary_outputs = model_D(history_len, kv_seq_len)
    if isinstance(rotary_outputs, dict):
        # Try to get outputs using expected names
        result_list = []
        for name in out_name_D:
            if name in rotary_outputs:
                result_list.append(rotary_outputs[name])
            else:
                # If expected key not found, use available keys
                available_keys = list(rotary_outputs.keys())
                if available_keys:
                    print(f"Warning: Expected output key '{name}' not found in model_D. Available keys: {available_keys}")
                    result_list = [rotary_outputs[k] for k in available_keys]
                    break
                else:
                    raise KeyError(f"Model D returned empty dict. Expected keys: {out_name_D}")
        rotary_outputs = result_list if result_list else [rotary_outputs[k] for k in list(rotary_outputs.keys())]
    elif not isinstance(rotary_outputs, (list, tuple)):
        rotary_outputs = [rotary_outputs]
else:
    rotary_outputs = model_E(history_len, kv_seq_len)
    if isinstance(rotary_outputs, dict):
        # Try to get outputs using expected names
        result_list = []
        for name in out_name_E:
            if name in rotary_outputs:
                result_list.append(rotary_outputs[name])
            else:
                # If expected key not found, use available keys
                available_keys = list(rotary_outputs.keys())
                if available_keys:
                    print(f"Warning: Expected output key '{name}' not found in model_E. Available keys: {available_keys}")
                    result_list = [rotary_outputs[k] for k in available_keys]
                    break
                else:
                    raise KeyError(f"Model E returned empty dict. Expected keys: {out_name_E}")
        rotary_outputs = result_list if result_list else [rotary_outputs[k] for k in list(rotary_outputs.keys())]
    elif not isinstance(rotary_outputs, (list, tuple)):
        rotary_outputs = [rotary_outputs]


input_feed_F = {
    in_name_F[num_keys_values]: kv_seq_len,
    in_name_F[num_keys_values_plus]: hidden_states,
    in_name_F[-2]: ids_len,
    in_name_F[-1]: attention_mask
}

# Prepare past keys and values for model F
# Note: Adjust based on how your model expects past keys/values
for i in range(num_layers):
    input_feed_F[in_name_F[i]] = past_keys_F  # May need indexing if model expects separate tensors
for i in range(num_layers, num_keys_values):
    input_feed_F[in_name_F[i]] = past_values_F  # May need indexing if model expects separate tensors
for i in range(rotary_outputs_len):
    input_feed_F[in_name_F[rotary_indices[i]]] = rotary_outputs[i]


print(f'\nTest Question: {query}\n\nQwenVL Answering:\n')
start_time = time.time()
while num_decode < max_single_chat_length:
    
    # Prepare inputs for model F - adjust based on actual model interface
    model_F_inputs = [input_feed_F[name] for name in in_name_F]
    all_outputs_F = model_F(*model_F_inputs)
    
    # Handle different output formats
    if isinstance(all_outputs_F, dict):
        all_outputs_F_list = [all_outputs_F[name] for name in out_name_F]
    elif isinstance(all_outputs_F, (list, tuple)):
        all_outputs_F_list = list(all_outputs_F)
    else:
        all_outputs_F_list = [all_outputs_F]
    
    # Get the last output (token IDs) - similar to ONNX version
    # all_outputs_F_list[-1] should be the token IDs output
    last_output = all_outputs_F_list[-1]
    
    # Convert to numpy to extract the token ID for decoding and stop check
    last_output_np = to_numpy(last_output)
    
    # Extract the token ID (handle different shapes)
    if last_output_np.ndim == 0:
        max_logit_ids = int(last_output_np)
    elif last_output_np.ndim == 1:
        max_logit_ids = int(last_output_np[0])
    else:
        max_logit_ids = int(last_output_np.flat[0])
    
    num_decode += 1
    
    if max_logit_ids in STOP_TOKEN:
        break
    
    if num_decode < 2:
        input_feed_F[in_name_F[-2]] = to_tensor(np.array([1], dtype=np.int64), dtype='int64')
        input_feed_F[in_name_F[-1]] = to_tensor(np.array([0], dtype=np.int8), dtype='int8')

    # Use the last output directly as input to model_A (like ONNX version)
    # Ensure it's in the correct format and shape (batch_size, seq_len = 1)
    # The output should be token IDs, reshape to (1, 1) if needed
    if hasattr(last_output, 'shape'):
        # If it's already a PaddlePaddle tensor
        if last_output.ndim == 0:
            # Scalar, reshape to (1, 1)
            next_token_input = to_tensor(np.array([[int(to_numpy(last_output))]], dtype=np.int32), dtype='int32')
        elif last_output.ndim == 1:
            # 1D array, reshape to (1, seq_len)
            last_output_np = to_numpy(last_output)
            next_token_input = to_tensor(last_output_np.reshape(1, -1), dtype='int32')
        elif last_output.ndim == 2:
            # Already 2D, use as is
            next_token_input = last_output
        else:
            # Flatten extra dimensions to (1, -1)
            last_output_np = to_numpy(last_output)
            next_token_input = to_tensor(last_output_np.reshape(1, -1), dtype='int32')
    else:
        # Convert to tensor
        last_output_np = np.array(last_output)
        if last_output_np.ndim == 0:
            next_token_input = to_tensor(np.array([[int(last_output_np)]], dtype=np.int32), dtype='int32')
        elif last_output_np.ndim == 1:
            next_token_input = to_tensor(last_output_np.reshape(1, -1), dtype='int32')
        elif last_output_np.ndim == 2:
            next_token_input = to_tensor(last_output_np, dtype='int32')
        else:
            next_token_input = to_tensor(last_output_np.reshape(1, -1), dtype='int32')
    
    next_hidden = model_A(next_token_input)
    if isinstance(next_hidden, dict):
        if out_name_A0 in next_hidden:
            next_hidden = next_hidden[out_name_A0]
        else:
            available_keys = list(next_hidden.keys())
            if available_keys:
                next_hidden = next_hidden[available_keys[0]]
            else:
                raise KeyError(f"Model A returned empty dict. Expected key: '{out_name_A0}'")
    input_feed_F[in_name_F[num_keys_values_plus]] = next_hidden
    
    if use_vision:
        rotary_outputs = model_D(input_feed_F[in_name_F[num_keys_values]], all_outputs_F_list[-2])
        if isinstance(rotary_outputs, dict):
            result_list = []
            for name in out_name_D:
                if name in rotary_outputs:
                    result_list.append(rotary_outputs[name])
                else:
                    available_keys = list(rotary_outputs.keys())
                    if available_keys:
                        result_list = [rotary_outputs[k] for k in available_keys]
                        break
            rotary_outputs = result_list if result_list else [rotary_outputs[k] for k in list(rotary_outputs.keys())]
        elif not isinstance(rotary_outputs, (list, tuple)):
            rotary_outputs = [rotary_outputs]
    else:
        rotary_outputs = model_E(input_feed_F[in_name_F[num_keys_values]], all_outputs_F_list[-2])
        if isinstance(rotary_outputs, dict):
            result_list = []
            for name in out_name_E:
                if name in rotary_outputs:
                    result_list.append(rotary_outputs[name])
                else:
                    available_keys = list(rotary_outputs.keys())
                    if available_keys:
                        result_list = [rotary_outputs[k] for k in available_keys]
                        break
            rotary_outputs = result_list if result_list else [rotary_outputs[k] for k in list(rotary_outputs.keys())]
        elif not isinstance(rotary_outputs, (list, tuple)):
            rotary_outputs = [rotary_outputs]

    for i in range(amount_of_outputs_F):
        input_feed_F[in_name_F[i]] = all_outputs_F_list[i]
    
    for i in range(rotary_outputs_len):
        input_feed_F[in_name_F[rotary_indices[i]]] = rotary_outputs[i]
    
    print(tokenizer.decode(max_logit_ids), end="", flush=True)

print(f"\n\nDecode: {(num_decode / (time.time() - start_time)):.3f} token/s")

