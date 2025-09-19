# Step 1
`uv pip install "optimum-onnx[onnxruntime]"@git+https://github.com/huggingface/optimum-onnx.git`

[doc](https://github.com/huggingface/optimum-onnx)

# Step 2

just use the CLI tool

`optimum-cli export onnx --model <model on HF> <name of new model>`