import os
import json
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTQuantizer, ORTModelForCausalLM
from optimum.onnxruntime.configuration import QuantizationConfig, AutoCalibrationConfig
from onnxruntime.quantization import QuantType # <-- Import QuantType
from datasets import Dataset

# 1. DEFINE YOUR PATHS
fp16_model_dir = "willcb/Qwen3-1.7B-Wordle"
quantized_model_dir = "exports/willcb-qwen3-1.7Bwordle-onnx-int8" 

os.makedirs(quantized_model_dir, exist_ok=True)

# 2. LOAD THE TOKENIZER
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(fp16_model_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# 3. PREPARE AND TOKENIZE THE CALIBRATION DATASET
print("Preparing and tokenizing calibration data...")
with open('data.json', 'r') as f:
    my_calibration_prompts = json.load(f)

def preprocess_function(examples):
    formatted_prompts = [
        tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
        for prompt in examples
    ]
    return tokenizer(formatted_prompts, padding="longest", max_length=512, truncation=True)

tokenized_prompts = preprocess_function(my_calibration_prompts)
# --- CORRECTED DATASET CREATION ---
tokenized_dataset = Dataset.from_dict(tokenized_prompts)
print(f"Created calibration dataset with {len(tokenized_dataset)} samples.")

# 4. DEFINE THE 8-BIT QUANTIZATION CONFIGURATION
print("Defining quantization configuration for maximum compatibility...")
qconfig = QuantizationConfig(
    is_static=True,
    format="QDQ",
    per_channel=True, # Recommended for better accuracy
    activations_dtype=QuantType.QUInt8, # Unsigned 8-bit for activations
    weights_dtype=QuantType.QInt8,       # Signed 8-bit for weights
)

# 5. INITIALIZE THE QUANTIZER
print("Loading ONNX model and initializing quantizer...")
onnx_model = ORTModelForCausalLM.from_pretrained(fp16_model_dir, export=True)
quantizer = ORTQuantizer.from_pretrained(onnx_model)

# 6. PERFORM THE TWO STEPS OF STATIC QUANTIZATION
print("Starting static quantization process...")

## STEP 1: CALIBRATION (`fit`)
print("Step 1: Running calibration (.fit) to compute tensor ranges...")
# For static quantization, we use a calibration configuration to collect the activation ranges
calibration_config = AutoCalibrationConfig.minmax(tokenized_dataset)
calibration_tensors_range = quantizer.fit(
    dataset=tokenized_dataset,
    calibration_config=calibration_config,
    operators_to_quantize=qconfig.operators_to_quantize,
    batch_size=1 
)
print("Calibration complete.")

## STEP 2: QUANTIZATION (`quantize`)
print("Step 2: Running quantization (.quantize) using the computed ranges...")
quantizer.quantize(
    quantization_config=qconfig,
    save_dir=quantized_model_dir,
    calibration_tensors_range=calibration_tensors_range,
    use_external_data_format=True
)
print(f"Quantization complete! Int8 model saved to: {quantized_model_dir}")

# 7. SAVE THE TOKENIZER FILES FOR A COMPLETE MODEL PACKAGE
tokenizer.save_pretrained(quantized_model_dir)
print("Tokenizer files saved.")