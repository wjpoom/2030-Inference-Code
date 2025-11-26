from transformers import AutoTokenizer
import mindspore as ms
from mindspore import Tensor, context
import numpy as np
import os
from PIL import Image
import time
import pickle
from onnx_wrappers import MindSporeWrapper  # 导入包装器类

def is_valid_image_path(image_path):
    if not os.path.exists(image_path):
        return False
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    _, ext = os.path.splitext(image_path)
    return ext.lower() in valid_extensions

# MindSpore settings
# 直接使用 CPU 版本（避免 GPU 错误信息）
import os
import sys
import warnings

# 抑制 MindSpore GPU 相关的错误和警告
os.environ['GLOG_minloglevel'] = '2'  # 抑制 C++ 日志
os.environ['MS_DISABLE_GPU'] = '1'  # 禁用 GPU 检查
warnings.filterwarnings('ignore', category=UserWarning)

# 重定向 stderr 以抑制 GPU 错误
class FilterStderr:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
    
    def write(self, text):
        # 过滤掉 GPU 相关的错误和警告信息
        if not any(keyword in text for keyword in ['libcuda.so', 'libcudnn.so', 'need by mindspore-gpu', '[ERROR] ME', '[WARNING] ME']):
            self.original_stderr.write(text)
    
    def flush(self):
        self.original_stderr.flush()

# 临时替换 stderr
_original_stderr = sys.stderr
sys.stderr = FilterStderr(sys.stderr)

# 直接使用 CPU
context.set_context(mode=context.GRAPH_MODE, device_target="CPU", device_id=0)
device_target = "CPU"
print("Using CPU for inference")

# 恢复 stderr
sys.stderr = _original_stderr

# Run the exported model by MindSpore
# NOTE: The slimm_mindspore directory has been created, but you need to convert ONNX models to MindSpore format (.mindir)
# You can use MindSpore's converter or export tools to convert from ONNX models in slimm_onnx/ directory
path = r'/mnt/public/users/chenyitong/projects/SliMM/ckpt/SliMM-Qwen2-0.5B-Tokenzier'
# Model paths - supports both .pkl (wrapper) and .mindir (native) formats
mindspore_model_A = r'/mnt/public/users/chenyitong/projects/SliMM/slimm_mindspore/SliMM_A.pkl'    # ONNX Runtime wrapper (created by create_wrappers.py)
mindspore_model_B = r'/mnt/public/users/chenyitong/projects/SliMM/slimm_mindspore/SliMM_B.pkl'
mindspore_model_C = r'/mnt/public/users/chenyitong/projects/SliMM/slimm_mindspore/SliMM_C.pkl'
mindspore_model_D = r'/mnt/public/users/chenyitong/projects/SliMM/slimm_mindspore/SliMM_D.pkl'
mindspore_model_E = r'/mnt/public/users/chenyitong/projects/SliMM/slimm_mindspore/SliMM_E.pkl'
mindspore_model_F = r'/mnt/public/users/chenyitong/projects/SliMM/slimm_mindspore_2/SliMM_F.pkl'  # ONNX Runtime wrapper
max_single_chat_length = 4096                         # It an adjustable value, but must less than max_seq_len.
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)


# Load MindSpore models
# Option 1: Load native MindSpore models (.mindir format)
# Option 2: Load ONNX Runtime wrappers (.pkl format)

def load_model(model_path):
    """加载模型，支持 .mindir 和 .pkl (wrapper) 格式"""
    if model_path.endswith('.pkl'):
        # 加载 ONNX Runtime 包装器
        # 需要导入包装器类才能正确反序列化
        with open(model_path, 'rb') as f:
            return pickle.load(f)
    else:
        # 加载原生 MindSpore 模型
        return ms.load(model_path)

try:
    # 尝试加载 .pkl 包装器文件
    model_A = load_model(mindspore_model_A)
    model_B = load_model(mindspore_model_B)
    model_C = load_model(mindspore_model_C)
    model_D = load_model(mindspore_model_D)
    model_E = load_model(mindspore_model_E)
    model_F = load_model(mindspore_model_F)
except FileNotFoundError:
    # 如果包装器不存在，尝试加载原生模型
    try:
        model_A = ms.load(mindspore_model_A)
        model_B = ms.load(mindspore_model_B)
        model_C = ms.load(mindspore_model_C)
        model_D = ms.load(mindspore_model_D)
        model_E = ms.load(mindspore_model_E)
        model_F = ms.load(mindspore_model_F)
    except:
        raise ValueError("Please run create_wrappers.py first to create model wrappers, or provide native MindSpore models")

# Get input/output names from loaded models
# 从实际加载的模型中获取输入输出名称
in_name_A0 = model_A.input_names[0] if hasattr(model_A, 'input_names') else "input_ids"
out_name_A0 = model_A.output_names[0] if hasattr(model_A, 'output_names') else "text_hidden_states"

in_name_B0 = model_B.input_names[0] if hasattr(model_B, 'input_names') else "pixel_values"
out_name_B0 = model_B.output_names[0] if hasattr(model_B, 'output_names') else "vision_hidden_states"

in_name_C0 = model_C.input_names[0] if hasattr(model_C, 'input_names') else "text_hidden_states"
in_name_C1 = model_C.input_names[1] if len(model_C.input_names) > 1 else "vision_hidden_states"
out_name_C0 = model_C.output_names[0] if hasattr(model_C, 'output_names') else "hidden_states"

in_name_D0 = model_D.input_names[0] if hasattr(model_D, 'input_names') else "history_len"
in_name_D1 = model_D.input_names[1] if len(model_D.input_names) > 1 else "kv_seq_len"
# 从模型 D 获取输出数量
rotary_outputs_len = len(model_D.output_names) if hasattr(model_D, 'output_names') else 2
out_name_D = model_D.output_names if hasattr(model_D, 'output_names') else [f"output_{i}" for i in range(rotary_outputs_len)]

in_name_E0 = model_E.input_names[0] if hasattr(model_E, 'input_names') else "history_len"
in_name_E1 = model_E.input_names[1] if len(model_E.input_names) > 1 else "kv_seq_len"
out_name_E = model_E.output_names if hasattr(model_E, 'output_names') else [f"output_{i}" for i in range(rotary_outputs_len)]

# For model F, get input/output names from the loaded model
in_name_F = model_F.input_names if hasattr(model_F, 'input_names') else []
out_name_F = model_F.output_names if hasattr(model_F, 'output_names') else []

# 从输出数量推断层数
amount_of_outputs_F = len(out_name_F)
num_layers = (amount_of_outputs_F - 2) // 2
num_keys_values = num_layers + num_layers
num_keys_values_plus = num_keys_values + 1
rotary_indices = np.arange(num_keys_values, num_keys_values + rotary_outputs_len, dtype=np.int32) + 2
amount_of_outputs_F -= 1

# 如果无法从模型获取名称，使用默认值
if not in_name_F:
    in_name_F = [f"input_{i}" for i in range(num_keys_values + rotary_outputs_len + 3)]
if not out_name_F:
    out_name_F = [f"output_{i}" for i in range(amount_of_outputs_F + 1)]

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
input_ids = Tensor(tokens, ms.int32)
ids_len = Tensor(np.array([tokens.shape[-1]], dtype=np.int64), ms.int64)
history_len = Tensor(np.array([0], dtype=np.int64), ms.int64)
kv_seq_len = Tensor(ids_len.asnumpy() + history_len.asnumpy(), ms.int64)
attention_mask = Tensor(np.array([1], dtype=np.int8), ms.int8)
max_single_chat_length -= tokens.shape[-1]

# Initialize past keys and values
# 从 model_F 的 ONNX session 中获取输入形状（与原始 ONNX 代码一致）
try:
    # 获取 model_F 的 session 来访问输入元数据
    session_F = model_F.session
    inputs_meta = session_F.get_inputs()
    
    # 按照原始 ONNX 代码的方式获取形状
    # past_keys_F shape: (inputs_meta[0].shape[0], 1, inputs_meta[0].shape[2], 0)
    # past_values_F shape: (inputs_meta[num_layers].shape[0], 1, 0, inputs_meta[num_layers].shape[3])
    key_input_shape = inputs_meta[0].shape
    value_input_shape = inputs_meta[num_layers].shape
    
    # 处理动态维度（None 或负数）
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
        0  # seq_len (初始为0)
    )
    
    # past_values_F: (batch, 1, 0, head_dim)
    past_values_shape = (
        get_shape_dim(value_input_shape, 0, 2),  # batch
        1,
        0,  # seq_len (初始为0)
        get_shape_dim(value_input_shape, 3, 64)  # head_dim
    )
    
    # 创建单个 past_key 和 past_value（所有层共享相同的形状）
    past_keys_F = Tensor(np.zeros(past_keys_shape, dtype=np.float32), ms.float32)
    past_values_F = Tensor(np.zeros(past_values_shape, dtype=np.float32), ms.float32)
    
    print(f"Initialized past_keys_F shape: {past_keys_shape}")
    print(f"Initialized past_values_F shape: {past_values_shape}")
    
except Exception as e:
    print(f"Warning: Could not infer shapes from model_F: {e}")
    print("Using default shapes - this may cause errors")
    # 默认形状（根据错误信息调整：batch=2, num_heads=32, head_dim=64）
    past_keys_shape = (2, 1, 32, 0)  # (batch=2, 1, num_heads=32, seq_len=0)
    past_values_shape = (2, 1, 0, 64)  # (batch=2, 1, seq_len=0, head_dim=64)
    past_keys_F = Tensor(np.zeros(past_keys_shape, dtype=np.float32), ms.float32)
    past_values_F = Tensor(np.zeros(past_values_shape, dtype=np.float32), ms.float32)


# Load input image
if is_valid_image_path(image_path):
    image = Image.open(image_path)
    image = image.resize((INPUT_IMAGE_SIZE[1], INPUT_IMAGE_SIZE[0]))
    if image.mode != 'RGB':
        image = image.convert('RGB')
    # 确保 pixel_values 是 uint8 类型
    # 直接从 PIL Image 转换为 uint8 numpy array
    image_array = np.array(image, dtype=np.uint8)
    pixel_values = np.transpose(image_array, (2, 0, 1))
    pixel_values = np.expand_dims(pixel_values, axis=0)
    # 强制确保是 uint8 类型
    pixel_values = pixel_values.astype(np.uint8)
    # 验证类型
    assert pixel_values.dtype == np.uint8, f"pixel_values dtype is {pixel_values.dtype}, expected uint8"
    use_vision = True
    print('\nChat with image.')
else:
    use_vision = False
    print('\nChat without image.')


# Start to run LLM
# Note: Adjust model calling based on your actual MindSpore model interface
# 包装器返回字典格式，键为输出名称
hidden_states_dict = model_A(input_ids)
# 包装器现在总是返回字典
hidden_states = hidden_states_dict[out_name_A0]


if use_vision:
    ids_len = Tensor(ids_len.asnumpy() + image_embed_size, ms.int64)
    kv_seq_len = Tensor(kv_seq_len.asnumpy() + image_embed_size, ms.int64)

    max_single_chat_length -= image_embed_size

    print('\nStart to Process the Image...')
    start_time = time.time()
    # 确保 pixel_values 是 uint8 类型（numpy array）
    # 注意：MindSpore 可能没有 uint8 类型，所以直接传递 numpy array
    if pixel_values.dtype != np.uint8:
        pixel_values = np.clip(pixel_values, 0, 255).astype(np.uint8)
    # 直接传递 numpy array，让包装器处理类型转换
    vision_hidden_states_dict = model_B(pixel_values)
    # 包装器返回字典格式
    vision_hidden_states = vision_hidden_states_dict[out_name_B0]
    print(f'\nImage Process Complete. Time Cost: {time.time() - start_time:.3f} Seconds')

    hidden_states_dict = model_C(hidden_states, vision_hidden_states)
    # 包装器返回字典格式
    hidden_states = hidden_states_dict[out_name_C0]

    rotary_outputs_dict = model_D(history_len, kv_seq_len)
    # 包装器返回字典，按输出名称顺序提取
    rotary_outputs = [rotary_outputs_dict[name] for name in out_name_D]
else:
    rotary_outputs_dict = model_E(history_len, kv_seq_len)
    # 包装器返回字典，按输出名称顺序提取
    rotary_outputs = [rotary_outputs_dict[name] for name in out_name_E]


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
    
    # Prepare inputs for model F - 使用字典格式传递输入
    all_outputs_F_dict = model_F(**input_feed_F)
    
    # 包装器返回字典，按输出名称顺序提取
    all_outputs_F_list = [all_outputs_F_dict[name] for name in out_name_F]
    
    # all_outputs_F_list[-1] 已经是 token_id（通过 argmax 得到），与原始 ONNX 代码一致
    # 原始代码: max_logit_ids = onnxruntime.OrtValue.numpy(all_outputs_F[-1])[0]
    # 注意：all_outputs_F[-1] 的形状是 (1, 1)，取 [0] 得到标量
    token_output = all_outputs_F_list[-1]
    if hasattr(token_output, 'asnumpy'):
        token_np = token_output.asnumpy()
    else:
        token_np = np.array(token_output)
    
    # 提取 token_id（形状可能是 (1, 1) 或 (1,)）
    if token_np.ndim == 0:
        max_logit_ids = int(token_np)
    elif token_np.ndim == 1:
        max_logit_ids = int(token_np[0])
    else:
        # 多维数组，取第一个元素
        max_logit_ids = int(token_np.flatten()[0])
    
    num_decode += 1
    
    if max_logit_ids in STOP_TOKEN:
        break
    
    if num_decode < 2:
        # 确保类型正确
        input_feed_F[in_name_F[-2]] = Tensor(np.array([1], dtype=np.int64), ms.int64)
        input_feed_F[in_name_F[-1]] = Tensor(np.array([0], dtype=np.int8), ms.int8)

    # 创建正确形状的 token: (batch_size=1, seq_len=1) = (1, 1)
    # 确保是 2 维数组，不是 3 维
    # 使用 squeeze 去除多余的维度
    next_token_np = np.array([[max_logit_ids]], dtype=np.int32)
    # 确保形状是 (1, 1)，去除任何多余的维度
    while len(next_token_np.shape) > 2:
        next_token_np = np.squeeze(next_token_np, axis=0)
    if len(next_token_np.shape) == 1:
        next_token_np = next_token_np.reshape(1, -1)
    
    next_token = Tensor(next_token_np, ms.int32)
    # 确保 next_token 是 2 维
    if len(next_token.shape) != 2:
        next_token = next_token.reshape(1, -1)
    
    # 注意：原始代码中 all_outputs_F[-1] 直接作为 input_ids 传递给 model_A
    # 但这里 all_outputs_F_list[-1] 已经是 token_id，所以需要创建新的 token
    next_hidden_dict = model_A(next_token)
    # 包装器返回字典格式
    next_hidden = next_hidden_dict[out_name_A0]
    input_feed_F[in_name_F[num_keys_values_plus]] = next_hidden
    
    if use_vision:
        # 确保类型正确：kv_seq_len 应该是 int64，all_outputs_F_list[-2] 应该是 int64
        kv_seq_len_input = input_feed_F[in_name_F[num_keys_values]]
        # 如果 kv_seq_len_input 不是 int64，转换它
        if hasattr(kv_seq_len_input, 'asnumpy'):
            kv_seq_len_np = kv_seq_len_input.asnumpy()
        else:
            kv_seq_len_np = np.array(kv_seq_len_input)
        # 确保是 int64 类型
        if kv_seq_len_np.dtype != np.int64:
            kv_seq_len_np = kv_seq_len_np.astype(np.int64)
        kv_seq_len_tensor = Tensor(kv_seq_len_np, ms.int64)
        
        # all_outputs_F_list[-2] 应该是 kv_seq_len 的输出，也应该是 int64
        kv_seq_len_output = all_outputs_F_list[-2]
        if hasattr(kv_seq_len_output, 'asnumpy'):
            kv_seq_len_out_np = kv_seq_len_output.asnumpy()
        else:
            kv_seq_len_out_np = np.array(kv_seq_len_output)
        # 确保是 int64 类型
        if kv_seq_len_out_np.dtype != np.int64:
            kv_seq_len_out_np = kv_seq_len_out_np.astype(np.int64)
        kv_seq_len_out_tensor = Tensor(kv_seq_len_out_np, ms.int64)
        
        rotary_outputs_dict = model_D(kv_seq_len_tensor, kv_seq_len_out_tensor)
        rotary_outputs = [rotary_outputs_dict[name] for name in out_name_D]
    else:
        # 确保类型正确
        kv_seq_len_input = input_feed_F[in_name_F[num_keys_values]]
        if hasattr(kv_seq_len_input, 'asnumpy'):
            kv_seq_len_np = kv_seq_len_input.asnumpy()
        else:
            kv_seq_len_np = np.array(kv_seq_len_input)
        if kv_seq_len_np.dtype != np.int64:
            kv_seq_len_np = kv_seq_len_np.astype(np.int64)
        kv_seq_len_tensor = Tensor(kv_seq_len_np, ms.int64)
        
        kv_seq_len_output = all_outputs_F_list[-2]
        if hasattr(kv_seq_len_output, 'asnumpy'):
            kv_seq_len_out_np = kv_seq_len_output.asnumpy()
        else:
            kv_seq_len_out_np = np.array(kv_seq_len_output)
        if kv_seq_len_out_np.dtype != np.int64:
            kv_seq_len_out_np = kv_seq_len_out_np.astype(np.int64)
        kv_seq_len_out_tensor = Tensor(kv_seq_len_out_np, ms.int64)
        
        rotary_outputs_dict = model_E(kv_seq_len_tensor, kv_seq_len_out_tensor)
        rotary_outputs = [rotary_outputs_dict[name] for name in out_name_E]

    for i in range(amount_of_outputs_F):
        input_feed_F[in_name_F[i]] = all_outputs_F_list[i]
    
    for i in range(rotary_outputs_len):
        input_feed_F[in_name_F[rotary_indices[i]]] = rotary_outputs[i]
    
    print(tokenizer.decode(max_logit_ids), end="", flush=True)

print(f"\n\nDecode: {(num_decode / (time.time() - start_time)):.3f} token/s")

