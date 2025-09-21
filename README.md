# ONNX export only (CLI)
## Step 1
`uv pip install "optimum-onnx[onnxruntime]"@git+https://github.com/huggingface/optimum-onnx.git`

[doc](https://github.com/huggingface/optimum-onnx)

## Step 2

just use the CLI tool

`optimum-cli export onnx --model <model on HF> <exports/name of new model>`

Note : not in the doc but you can use --dtype fp16 to avoid default export in fp32. --dtype bf16 is accepted however it is not actually handled by the export script  

# ONNX export and quantization
define original (source) model in `quantize.py`
create dataset for calibration, representative of the prompts seen by the model

WARNING : still looking into it but static quantization requires a LOT of VRAM. like >150GB. Dynamic is more reasonable but still >30GB. Also quantization tools will produce weight files with .onnx.data extensions whereas transformers.js expects .onnx_data extensions. You can't just change the name because model.onnx references model.onnx.data. Use the `sanitize.py` script to change the location in the model.onnx file.  
`python sanitize.py exports/<model_folder>/model_int8.onnx <updated_model_folder>/model_int8`