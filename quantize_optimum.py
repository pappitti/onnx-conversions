import os
import json
import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTQuantizer, ORTModelForCausalLM
from optimum.onnxruntime.configuration import QuantizationConfig, AutoCalibrationConfig, CalibrationConfig, QuantizationMode
from onnxruntime.quantization import QuantType # <-- Import QuantType
from datasets import Dataset

class ExternalDataCalibrationConfig(CalibrationConfig):
    # We must match the parent method's signature exactly
    def create_calibrator(
        self,
        onnx_model_path,
        operators_to_quantize,
        use_external_data_format: bool = False, # This will be False (default) from the caller
        force_symmetric_range: bool = False,
        augmented_model_name: str = "augmented_model.onnx",
    ):
        # 2. Call the parent method, but FORCE use_external_data_format to be True
        return super().create_calibrator(
            onnx_model_path=onnx_model_path,
            operators_to_quantize=operators_to_quantize,
            use_external_data_format=True,  # This is the crucial override
            force_symmetric_range=force_symmetric_range,
            augmented_model_name=augmented_model_name,
        )

# 1. DEFINE YOUR PATHS
fp16_model_dir = "willcb/Qwen3-1.7B-Wordle"
quantized_model_dir = "exports/willcb-qwen3-1.7B-wordle-onnx-int8" 

os.makedirs(quantized_model_dir, exist_ok=True)

# 2. LOAD THE TOKENIZER
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(fp16_model_dir)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

calibration_tensors_range_file = None #"1495f5fc-9702-11f0-b384-de2fc79e6085.data"

if not calibration_tensors_range_file:

    # 3. PREPARE AND TOKENIZE THE CALIBRATION DATASET
    print("Preparing and tokenizing calibration data...")
    with open('data.json', 'r') as f:
        my_calibration_prompts = json.load(f)

    def preprocess_function(examples):
        formatted_prompts = [
            tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            for prompt in examples
        ]
        tokenized_inputs = tokenizer(
            formatted_prompts,
            padding="longest",
            max_length=2048,
            truncation=True,
            return_tensors="np" # Return numpy arrays
        )

        # Manually create position_ids
        input_ids = tokenized_inputs['input_ids']
        shape = input_ids.shape
        position_ids = np.arange(0, shape[1], dtype=np.int64).reshape(1, shape[1])
        tokenized_inputs['position_ids'] = np.repeat(position_ids, shape[0], axis=0)

        # Convert numpy arrays back to lists for Dataset creation
        return {k: v.tolist() for k, v in tokenized_inputs.items()}

    tokenized_prompts = preprocess_function(my_calibration_prompts)
    # --- CORRECTED DATASET CREATION ---
    tokenized_dataset = Dataset.from_dict(tokenized_prompts)
    print(f"Created calibration dataset with {len(tokenized_dataset)} samples.")

# 4. DEFINE THE 8-BIT QUANTIZATION CONFIGURATION
print("Defining quantization configuration for maximum compatibility...")
qconfig = QuantizationConfig(
    is_static=False,  # Dynamic quantization for broader operator support
    mode = QuantizationMode.IntegerOps, # For Dynamic quantization  (QLinearOps is for static)
    format="QDQ",
    per_channel=True, # Recommended for better accuracy
    activations_dtype=QuantType.QUInt8, # Unsigned 8-bit for activations
    weights_dtype=QuantType.QInt8,       # Signed 8-bit for weights
)

# 5. INITIALIZE THE QUANTIZER
print("Loading ORIGINAL model and initializing quantizer...")
onnx_model = ORTModelForCausalLM.from_pretrained(
    fp16_model_dir, 
    export=True,
    use_cache=True, 
    use_io_binding=False)
quantizer = ORTQuantizer.from_pretrained(onnx_model)

# 6. PERFORM THE TWO STEPS OF STATIC QUANTIZATION
print("Starting static quantization process...")

if calibration_tensors_range_file:
    print(f"Loading precomputed calibration ranges from {calibration_tensors_range_file}...")
    calibration_tensors_range = np.load(calibration_tensors_range_file, allow_pickle=True).item()
    print("Loaded calibration ranges.")

elif qconfig.is_static:
    ## STEP 1: CALIBRATION (`fit`)
    print("Step 1: Running calibration (.fit) to compute tensor ranges...")
    # For static quantization, we use a calibration configuration to collect the activation ranges
    base_config = AutoCalibrationConfig.minmax(tokenized_dataset)

    # 4. Now, create an instance of our custom class using the settings from the base config
    calibration_config = ExternalDataCalibrationConfig(
        dataset_name=base_config.dataset_name,
        dataset_config_name=base_config.dataset_config_name,
        dataset_split=base_config.dataset_split,
        dataset_num_samples=base_config.dataset_num_samples,
        method=base_config.method,
    )

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
    calibration_tensors_range=calibration_tensors_range if qconfig.is_static else None,
    use_external_data_format=True,
    file_suffix="int8"
)
print(f"Quantization complete! Int8 model saved to: {quantized_model_dir}")

# 7. SAVE THE TOKENIZER FILES FOR A COMPLETE MODEL PACKAGE
tokenizer.save_pretrained(quantized_model_dir)
print("Tokenizer files saved.")