from transformers import AutoTokenizer
import onnxruntime
import numpy as np
import os
from PIL import Image
import time

def is_valid_image_path(image_path):
    if not os.path.exists(image_path):
        return False
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    _, ext = os.path.splitext(image_path)
    return ext.lower() in valid_extensions

# Run the exported model by ONNX Runtime
path = r'/mnt/public/users/chenyitong/projects/SliMM/ckpt/SliMM-Qwen2-0.5B-Tokenzier'    
onnx_model_A = r'/mnt/public/users/chenyitong/projects/SliMM/slimm_onnx/SliMM_A.onnx'    # Assign a path where the exported QwenVL model stored.
onnx_model_B = r'/mnt/public/users/chenyitong/projects/SliMM/slimm_onnx/SliMM_B.onnx'
onnx_model_C = r'/mnt/public/users/chenyitong/projects/SliMM/slimm_onnx/SliMM_C.onnx'
onnx_model_D = r'/mnt/public/users/chenyitong/projects/SliMM/slimm_onnx/SliMM_D.onnx'
onnx_model_E = r'/mnt/public/users/chenyitong/projects/SliMM/slimm_onnx/SliMM_E.onnx'
onnx_model_F = r'/mnt/public/users/chenyitong/projects/SliMM/slimm_onnx_2/SliMM_F.onnx'  
max_single_chat_length = 4096                         # It an adjustable value, but must less than max_seq_len.
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)


# ONNX Runtime settings
session_opts = onnxruntime.SessionOptions()
session_opts.log_severity_level = 4                   # fatal level = 4, it an adjustable value.
session_opts.log_verbosity_level = 4                  # fatal level = 4, it an adjustable value.
session_opts.inter_op_num_threads = 0                 # Run different nodes with num_threads. Set 0 for auto.
session_opts.intra_op_num_threads = 0                 # Under the node, execute the operators with num_threads. Set 0 for auto.
session_opts.enable_cpu_mem_arena = True              # True for execute speed; False for less memory usage.
session_opts.execution_mode = onnxruntime.ExecutionMode.ORT_SEQUENTIAL
session_opts.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
session_opts.add_session_config_entry("session.set_denormal_as_zero", "1")
session_opts.add_session_config_entry("session.intra_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.inter_op.allow_spinning", "1")
session_opts.add_session_config_entry("session.enable_quant_qdq_cleanup", "1")
session_opts.add_session_config_entry("session.qdq_matmulnbits_accuracy_level", "4")
session_opts.add_session_config_entry("optimization.enable_gelu_approximation", "1")
session_opts.add_session_config_entry("disable_synchronize_execution_providers", "1")
session_opts.add_session_config_entry("optimization.minimal_build_optimizations", "")
session_opts.add_session_config_entry("session.use_device_allocator_for_initializers", "1")

device_type = 'cpu'
device_id = 0


ort_session_A = onnxruntime.InferenceSession(onnx_model_A, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_A = ort_session_A.get_inputs()
out_name_A = ort_session_A.get_outputs()
in_name_A0 = in_name_A[0].name
out_name_A0 = out_name_A[0].name


ort_session_B = onnxruntime.InferenceSession(onnx_model_B, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_B = ort_session_B.get_inputs()
out_name_B = ort_session_B.get_outputs()
in_name_B0 = in_name_B[0].name
out_name_B0 = out_name_B[0].name


ort_session_C = onnxruntime.InferenceSession(onnx_model_C, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_C = ort_session_C.get_inputs()
out_name_C = ort_session_C.get_outputs()
in_name_C0 = in_name_C[0].name
in_name_C1 = in_name_C[1].name
out_name_C0 = out_name_C[0].name


ort_session_D = onnxruntime.InferenceSession(onnx_model_D, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_D = ort_session_D.get_inputs()
out_name_D = ort_session_D.get_outputs()
in_name_D0 = in_name_D[0].name
in_name_D1 = in_name_D[1].name
rotary_outputs_len = len(out_name_D)
out_name_D = [out_name_D[i].name for i in range(rotary_outputs_len)]


ort_session_E = onnxruntime.InferenceSession(onnx_model_E, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_E = ort_session_E.get_inputs()
out_name_E = ort_session_E.get_outputs()
in_name_E0 = in_name_E[0].name
in_name_E1 = in_name_E[1].name
out_name_E = [out_name_E[i].name for i in range(rotary_outputs_len)]


ort_session_F = onnxruntime.InferenceSession(onnx_model_F, sess_options=session_opts, providers=['CPUExecutionProvider'])
in_name_F = ort_session_F.get_inputs()
out_name_F = ort_session_F.get_outputs()
amount_of_outputs_F = len(out_name_F)
in_name_F = [in_name_F[i].name for i in range(len(in_name_F))]
out_name_F = [out_name_F[i].name for i in range(amount_of_outputs_F)]


num_layers = (amount_of_outputs_F - 2) // 2
num_keys_values = num_layers + num_layers
num_keys_values_plus = num_keys_values + 1
rotary_indices = np.arange(num_keys_values, num_keys_values + rotary_outputs_len, dtype=np.int32) + 2
amount_of_outputs_F -= 1

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
input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device_type, device_id)
ids_len = np.array([tokens.shape[-1]], dtype=np.int64)
history_len = np.array([0], dtype=np.int64)
kv_seq_len = onnxruntime.OrtValue.ortvalue_from_numpy(ids_len + history_len, device_type, device_id)
ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(ids_len, device_type, device_id)
history_len = onnxruntime.OrtValue.ortvalue_from_numpy(history_len, device_type, device_id)
attention_mask = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int8), device_type, device_id)
max_single_chat_length -= tokens.shape[-1]
if device_type != 'dml':
    past_keys_F = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_F._inputs_meta[0].shape[0], 1, ort_session_F._inputs_meta[0].shape[2], 0), dtype=np.float32), device_type, device_id)
    past_values_F = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_F._inputs_meta[num_layers].shape[0], 1, 0, ort_session_F._inputs_meta[num_layers].shape[3]), dtype=np.float32), device_type, device_id)
else:
    # Crash with unknown reason.
    past_keys_F = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_F._inputs_meta[0].shape[0], 1, ort_session_F._inputs_meta[0].shape[2], 0), dtype=np.float32), 'cpu', device_id)
    past_values_F = onnxruntime.OrtValue.ortvalue_from_numpy(np.zeros((ort_session_F._inputs_meta[num_layers].shape[0], 1, 0, ort_session_F._inputs_meta[num_layers].shape[3]), dtype=np.float32), 'cpu', device_id)


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
hidden_states = ort_session_A.run_with_ort_values([out_name_A0], {in_name_A0: input_ids})[0]


if use_vision:
    ids_len = onnxruntime.OrtValue.ortvalue_from_numpy(onnxruntime.OrtValue.numpy(ids_len) + image_embed_size, device_type, device_id)
    kv_seq_len = onnxruntime.OrtValue.ortvalue_from_numpy(onnxruntime.OrtValue.numpy(kv_seq_len) + image_embed_size, device_type, device_id)

    max_single_chat_length -= image_embed_size

    print('\nStart to Process the Image...')
    start_time = time.time()
    vision_hidden_states = ort_session_B.run_with_ort_values([out_name_B0], {in_name_B0: onnxruntime.OrtValue.ortvalue_from_numpy(pixel_values, device_type, device_id)})[0]
    print(f'\nImage Process Complete. Time Cost: {time.time() - start_time:.3f} Seconds')

    hidden_states = ort_session_C.run_with_ort_values([out_name_C0], {in_name_C0: hidden_states, in_name_C1: vision_hidden_states})[0]

    rotary_outputs = ort_session_D.run_with_ort_values(out_name_D, {in_name_D0: history_len, in_name_D1: kv_seq_len})
else:
    rotary_outputs = ort_session_E.run_with_ort_values(out_name_E, {in_name_E0: history_len, in_name_E1: kv_seq_len})


input_feed_F = {
    in_name_F[num_keys_values]: kv_seq_len,
    in_name_F[num_keys_values_plus]: hidden_states,
    in_name_F[-2]: ids_len,
    in_name_F[-1]: attention_mask
}

for i in range(num_layers):
    input_feed_F[in_name_F[i]] = past_keys_F
for i in range(num_layers, num_keys_values):
    input_feed_F[in_name_F[i]] = past_values_F
for i in range(rotary_outputs_len):
    input_feed_F[in_name_F[rotary_indices[i]]] = rotary_outputs[i]


print(f'\nTest Question: {query}\n\nQwenVL Answering:\n')
start_time = time.time()
while num_decode < max_single_chat_length:
    
    all_outputs_F = ort_session_F.run_with_ort_values(out_name_F, input_feed_F)
    
    max_logit_ids = onnxruntime.OrtValue.numpy(all_outputs_F[-1])[0]
    num_decode += 1
    
    if max_logit_ids in STOP_TOKEN:
        break    
        
    if num_decode < 2:
        input_feed_F[in_name_F[-2]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([1], dtype=np.int64), device_type, device_id)
        input_feed_F[in_name_F[-1]] = onnxruntime.OrtValue.ortvalue_from_numpy(np.array([0], dtype=np.int8), device_type, device_id)

    input_feed_F[in_name_F[num_keys_values_plus]] = ort_session_A.run_with_ort_values([out_name_A0], {in_name_A0: all_outputs_F[-1]})[0]
    
    if use_vision:
        rotary_outputs = ort_session_D.run_with_ort_values(out_name_D, {in_name_D0: input_feed_F[in_name_F[num_keys_values]], in_name_D1: all_outputs_F[-2]})
    else:
        rotary_outputs = ort_session_E.run_with_ort_values(out_name_E, {in_name_E0: input_feed_F[in_name_F[num_keys_values]], in_name_E1: all_outputs_F[-2]})

    for i in range(amount_of_outputs_F):
        input_feed_F[in_name_F[i]] = all_outputs_F[i]
        
    for i in range(rotary_outputs_len):
        input_feed_F[in_name_F[rotary_indices[i]]] = rotary_outputs[i]
        
    print(tokenizer.decode(max_logit_ids), end="", flush=True)
    
print(f"\n\nDecode: {(num_decode / (time.time() - start_time)):.3f} token/s")