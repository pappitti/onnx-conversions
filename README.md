# Step 1
`uv pip install "optimum-onnx[onnxruntime]"@git+https://github.com/huggingface/optimum-onnx.git`

[doc](https://github.com/huggingface/optimum-onnx)

# Step 2

just use the CLI tool

`optimum-cli export onnx --model <model on HF> <exports/name of new model>`

Note : not in the doc but you can use --dtype fp16 to avoid default export in fp32. --dtype bf16 is accepted however it is not actually handled by the export script