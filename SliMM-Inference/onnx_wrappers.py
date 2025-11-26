"""
ONNX Runtime 包装器类
用于在 MindSpore、PaddlePaddle 和 JAX 中使用 ONNX 模型
"""

import onnxruntime as ort
import numpy as np
import warnings


def get_available_providers(use_gpu=False):
    """获取可用的 ONNX Runtime providers，避免警告"""
    available = ort.get_available_providers()
    if use_gpu and 'CUDAExecutionProvider' in available:
        return ['CUDAExecutionProvider', 'CPUExecutionProvider']
    else:
        return ['CPUExecutionProvider']


class ONNXWrapper:
    """通用的 ONNX Runtime 包装器基类"""
    def __init__(self, onnx_path, use_gpu=False):
        self.onnx_path = onnx_path
        self.use_gpu = use_gpu
        self._session = None
        # 预先获取输入输出名称（这些可以序列化）
        # 只使用可用的 providers，避免警告
        providers = get_available_providers(use_gpu)
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            temp_session = ort.InferenceSession(onnx_path, providers=providers)
        self.input_names = [inp.name for inp in temp_session.get_inputs()]
        self.output_names = [out.name for out in temp_session.get_outputs()]
        del temp_session
    
    @property
    def session(self):
        """延迟加载 session（因为无法序列化）"""
        if self._session is None:
            # 只使用可用的 providers，避免警告
            providers = get_available_providers(self.use_gpu)
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=UserWarning)
                self._session = ort.InferenceSession(self.onnx_path, providers=providers)
        return self._session
    
    def _to_numpy(self, tensor):
        """将框架张量转换为 numpy"""
        if isinstance(tensor, np.ndarray):
            # 如果是 numpy array，直接返回（保持原始类型）
            return tensor
        # 尝试各种框架的转换方法
        if hasattr(tensor, 'asnumpy'):  # MindSpore
            result = tensor.asnumpy()
            # 如果结果是 numpy array，直接返回
            if isinstance(result, np.ndarray):
                return result
            return np.array(result)
        if hasattr(tensor, 'numpy'):  # PaddlePaddle, TensorFlow
            result = tensor.numpy()
            if isinstance(result, np.ndarray):
                return result
            return np.array(result)
        if hasattr(tensor, '__array__'):  # JAX
            return np.array(tensor)
        return np.array(tensor)
    
    def _get_expected_dtype(self, input_name):
        """根据输入名称获取期望的数据类型"""
        # 从 session 获取输入信息
        session = self.session
        for inp in session.get_inputs():
            if inp.name == input_name:
                # 返回 ONNX 类型对应的 numpy 类型
                # 注意：需要检查完整的类型字符串
                type_str = str(inp.type).lower()
                if 'int64' in type_str:
                    return np.int64
                elif 'int32' in type_str:
                    return np.int32
                elif 'uint8' in type_str:
                    return np.uint8
                elif 'int8' in type_str:
                    return np.int8
                elif 'float32' in type_str or 'float' in type_str:
                    return np.float32
        return None
    
    def _from_numpy(self, array, framework='numpy'):
        """将 numpy 转换为框架张量（在子类中实现）"""
        return array
    
    def __call__(self, *args, **kwargs):
        """执行推理"""
        # 准备输入
        inputs = {}
        
        # 处理位置参数
        for i, arg in enumerate(args):
            if i < len(self.input_names):
                input_name = self.input_names[i]
                val_np = self._to_numpy(arg)
                # 检查并转换类型
                expected_dtype = self._get_expected_dtype(input_name)
                if expected_dtype:
                    # 确保类型转换正确，特别是 uint8
                    if val_np.dtype != expected_dtype:
                        # 对于 uint8，需要特别处理，因为某些框架可能转换为 int8
                        if expected_dtype == np.uint8:
                            # 如果当前是 int8，需要特殊处理
                            if val_np.dtype == np.int8:
                                # int8 范围是 -128 到 127，需要转换为 0-255
                                # 方法：将 -128~-1 映射到 128~255，0~127 保持不变
                                # 使用 np.where 处理负数，先转为 int16 避免溢出
                                val_np = np.where(val_np < 0, val_np.astype(np.int16) + 256, val_np.astype(np.int16)).astype(np.uint8)
                            elif val_np.dtype in [np.int16, np.int32, np.int64]:
                                # 将 signed 转换为 unsigned
                                val_np = np.clip(val_np, 0, 255).astype(np.uint8)
                            else:
                                # 其他类型直接转换
                                val_np = np.clip(val_np, 0, 255).astype(np.uint8)
                        else:
                            val_np = val_np.astype(expected_dtype)
                    # 最终验证类型 - 强制确保类型正确
                    if val_np.dtype != expected_dtype:
                        # 强制转换
                        val_np = val_np.astype(expected_dtype)
                # 最终检查：确保类型完全匹配（不抛出异常，只打印警告）
                if expected_dtype and val_np.dtype != expected_dtype:
                    import warnings
                    warnings.warn(f"Type mismatch for {input_name}: got {val_np.dtype}, expected {expected_dtype}. Forcing conversion.")
                    val_np = val_np.astype(expected_dtype)
                inputs[input_name] = val_np
        
        # 处理关键字参数
        for key, value in kwargs.items():
            if key in self.input_names:
                val_np = self._to_numpy(value)
                # 检查并转换类型
                expected_dtype = self._get_expected_dtype(key)
                if expected_dtype:
                    # 确保类型转换正确，特别是 uint8
                    if val_np.dtype != expected_dtype:
                        # 对于 uint8，需要特别处理
                        if expected_dtype == np.uint8:
                            # 如果当前是 int8，需要特殊处理
                            if val_np.dtype == np.int8:
                                # int8 范围是 -128 到 127，需要转换为 0-255
                                # 方法：将 -128~-1 映射到 128~255，0~127 保持不变
                                # 使用 np.where 处理负数，先转为 int16 避免溢出
                                val_np = np.where(val_np < 0, val_np.astype(np.int16) + 256, val_np.astype(np.int16)).astype(np.uint8)
                            elif val_np.dtype in [np.int16, np.int32, np.int64]:
                                # 将 signed 转换为 unsigned
                                val_np = np.clip(val_np, 0, 255).astype(np.uint8)
                            else:
                                # 其他类型直接转换
                                val_np = np.clip(val_np, 0, 255).astype(np.uint8)
                        else:
                            val_np = val_np.astype(expected_dtype)
                    # 最终验证类型 - 强制确保类型正确
                    if val_np.dtype != expected_dtype:
                        # 强制转换
                        val_np = val_np.astype(expected_dtype)
                # 最终检查：确保类型完全匹配（不抛出异常，只打印警告）
                if expected_dtype and val_np.dtype != expected_dtype:
                    import warnings
                    warnings.warn(f"Type mismatch for {key}: got {val_np.dtype}, expected {expected_dtype}. Forcing conversion.")
                    val_np = val_np.astype(expected_dtype)
                inputs[key] = val_np
        
        # 运行推理
        outputs = self.session.run(None, inputs)
        
        # 转换输出并返回字典格式（兼容 ONNX 输出名称）
        # 始终返回字典，即使只有一个输出
        output_dict = {}
        for i, out_name in enumerate(self.output_names):
            output_dict[out_name] = self._from_numpy(outputs[i])
        
        # 始终返回字典格式，方便统一处理
        return output_dict
    
    def __getstate__(self):
        """序列化时只保存路径和名称"""
        return {
            'onnx_path': self.onnx_path,
            'input_names': self.input_names,
            'output_names': self.output_names,
            'use_gpu': self.use_gpu
        }
    
    def __setstate__(self, state):
        """反序列化时恢复"""
        self.onnx_path = state['onnx_path']
        self.input_names = state['input_names']
        self.output_names = state['output_names']
        self.use_gpu = state.get('use_gpu', False)
        self._session = None


class MindSporeWrapper(ONNXWrapper):
    """MindSpore 包装器"""
    def _from_numpy(self, array):
        try:
            import mindspore as ms
            import mindspore.common.dtype as mstype
            # 根据数组的实际类型选择合适的 MindSpore 类型
            if isinstance(array, np.ndarray):
                if array.dtype == np.int64:
                    return ms.Tensor(array, dtype=mstype.int64)
                elif array.dtype == np.int32:
                    return ms.Tensor(array, dtype=mstype.int32)
                elif array.dtype == np.int8:
                    return ms.Tensor(array, dtype=mstype.int8)
                elif array.dtype == np.uint8:
                    return ms.Tensor(array, dtype=mstype.uint8)
                else:
                    # 默认使用 float32
                    return ms.Tensor(array, dtype=mstype.float32)
            else:
                return ms.Tensor(array, dtype=mstype.float32)
        except ImportError:
            # 如果 MindSpore 未安装，返回 numpy
            return array
        except Exception as e:
            # 如果类型转换失败，尝试使用 float32
            try:
                import mindspore as ms
                return ms.Tensor(np.array(array, dtype=np.float32), dtype=ms.float32)
            except:
                return array


class PaddlePaddleWrapper(ONNXWrapper):
    """PaddlePaddle 包装器"""
    def _from_numpy(self, array):
        try:
            import paddle
            # 确保数据是 numpy array
            if not isinstance(array, np.ndarray):
                array = np.array(array)
            
            # 确定目标 dtype（保持原始 dtype，而不是总是转换为 float32）
            # 将 numpy dtype 转换为 PaddlePaddle dtype 字符串
            dtype_map = {
                np.int32: 'int32',
                np.int64: 'int64',
                np.int8: 'int8',
                np.uint8: 'uint8',
                np.float32: 'float32',
                np.float64: 'float64',
            }
            target_dtype = dtype_map.get(array.dtype, 'float32')
            
            # 尝试使用 paddle.to_tensor (PaddlePaddle 2.0+)
            if hasattr(paddle, 'to_tensor'):
                try:
                    return paddle.to_tensor(array, dtype=target_dtype)
                except:
                    pass
            
            # 回退方案：使用 paddle.Tensor 或 paddle.cast
            try:
                tensor = paddle.Tensor(array)
                if hasattr(paddle, 'cast') and target_dtype != 'float32':
                    return paddle.cast(tensor, dtype=target_dtype)
                return tensor
            except:
                # 最后的回退：直接使用 numpy array（如果模型接受）
                return array
        except ImportError:
            # 如果 PaddlePaddle 未安装，返回 numpy
            return array


class JAXWrapper(ONNXWrapper):
    """JAX 包装器"""
    def _from_numpy(self, array):
        try:
            import jax.numpy as jnp
            return jnp.array(array)
        except ImportError:
            # 如果 JAX 未安装，返回 numpy
            return array

