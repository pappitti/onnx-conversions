import os
import onnx
from pathlib import Path
from onnxruntime.transformers.optimizer import optimize_model
from onnxruntime.quantization import (
    matmul_nbits_quantizer,  # onnxruntime >= 1.22.0
    quant_utils
)

# Path Setting
original_folder_path = "exports/willcb-qwen3-1.7B-wordle-onnx-fp32"              # The original folder.
quanted_folder_path = "exports/willcb-qwen3-1.7B-wordle-onnx-q4"                 # The optimized folder.
model_path = os.path.join(original_folder_path, "Model.onnx")                    # The original fp32 model path.
quanted_model_path = os.path.join(quanted_folder_path, "Model_Optimized.onnx")   # The optimized model stored path.


algorithm = "DEFAULT"                                                            # ["DEFAULT", "RTN", "HQQ",], HQQ will very slow both in quant and inference.
bits = 4                                                                         # [4, 8]
op_types = ["MatMul"]                                                            # ["MatMul", "Gather"]; Adding Gather may get errors.
quant_axes = [0]                                                                 # Target axes to quant the quant data.
block_size = 128                                                                 # [32, 64, 128, 256]; A smaller block_size yields greater accuracy but increases quantization time and model size.
accuracy_level = 4                                                               # 0:default, 1:fp32, 2:fp16, 3:bf16, 4:int8
quant_symmetric = False                                                          # False may get more accuracy.
nodes_to_exclude = None                                                          # Set the node names here. Such as: ["/layers.0/mlp/down_proj/MatMul"]


# Start Weight-Only Quantize
model = quant_utils.load_model_with_shape_infer(Path(model_path))

if algorithm == "RTN":
    quant_config = matmul_nbits_quantizer.RTNWeightOnlyQuantConfig(
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=tuple(op_types)
    )
elif algorithm == "HQQ":
    quant_config = matmul_nbits_quantizer.HQQWeightOnlyQuantConfig(
        bits=bits,
        block_size=block_size,
        axis=quant_axes[0],
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=tuple(op_types),
        quant_axes=tuple((op_types[i], quant_axes[i]) for i in range(len(op_types)))
    )
else:
    quant_config = matmul_nbits_quantizer.DefaultWeightOnlyQuantConfig(
        block_size=block_size,
        is_symmetric=quant_symmetric,
        accuracy_level=accuracy_level,
        quant_format=quant_utils.QuantFormat.QOperator,
        op_types_to_quantize=tuple(op_types),
        quant_axes=tuple((op_types[i], quant_axes[i]) for i in range(len(op_types)))
    )
quant_config.bits = bits
quant = matmul_nbits_quantizer.MatMulNBitsQuantizer(
    model,
    block_size=block_size,
    is_symmetric=quant_symmetric,
    accuracy_level=accuracy_level,
    quant_format=quant_utils.QuantFormat.QOperator,
    op_types_to_quantize=tuple(op_types),
    quant_axes=tuple((op_types[i], quant_axes[i]) for i in range(len(op_types))),
    algo_config=quant_config,
    nodes_to_exclude=nodes_to_exclude
)
quant.process()

os.makedirs(quanted_folder_path, exist_ok=True)

quant.model.save_model_to_file(
    quanted_model_path,
    True                                         # save_as_external_data
)