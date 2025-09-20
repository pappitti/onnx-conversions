from transformers import pipeline
from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer
model = ORTModelForCausalLM.from_pretrained("PITTI/willcb-Qwen3-1.7B-Wordle-onnx")
tokenizer = AutoTokenizer.from_pretrained("PITTI/willcb-Qwen3-1.7B-Wordle-onnx")
onnx_pipeline = pipeline("text-generation", model=model, tokenizer=tokenizer)
messages = [
        {
            "content": "You are a competitive game player. Make sure you read the game instructions carefully, and always follow the required format.\n\nIn each turn, think step-by-step inside <think>...</think> tags, then follow the instructions inside <guess>...</guess> tags.",
            "role": "system"
        },
        {
            "content": "You are Player 0 in Wordle.\nA secret 5-letter word has been chosen. You have 6 attempts to guess it.\nFor each guess, wrap your word in square brackets (e.g., [apple]).\nFeedback for each letter will be given as follows:\n - G (green): correct letter in the correct position\n - Y (yellow): letter exists in the word but in the wrong position\n - X (wrong): letter is not in the word\nEnter your guess to begin.\n",
            "role": "user"
        }
    ]
response = onnx_pipeline(messages, max_new_tokens=256, do_sample=False, top_p=0.7, temperature=0.1, repetition_penalty=1.1, num_return_sequences=1)
print(response)