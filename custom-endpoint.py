# main.py
import torch
import gc
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
from jinja2 import Template
import json

MODEL_NAME = "Qwen/Qwen3-4B-Instruct-2507" # change this as needed 
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto").to(DEVICE)
chat_template = getattr(tokenizer, "chat_template", None)

app = FastAPI()

class ChatRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = 5000
    stream: bool = False

def build_prompt(messages):
    if chat_template:
        template = Template(chat_template)
        return template.render(messages=messages, tools=[], add_generation_prompt=True)
    else:
        prompt = ""
        for msg in messages:
            prompt += f"### {msg['role'].capitalize()}:\n{msg['content']}\n\n"
        prompt += "### Assistant:\n"
        return prompt

def generate_response(prompt, max_new_tokens=512):
    inputs = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            top_p=0.9,
            temperature=0.7,
        )
    decoded = tokenizer.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
    del inputs, outputs
    torch.cuda.empty_cache()
    gc.collect()
    return decoded

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatRequest):
    # Build standard chat prompt
    prompt = build_prompt(req.messages)
    # Streaming not implemented for now (OpenAI benchmark script does not stream)
    response_text = generate_response(prompt, max_new_tokens=req.max_tokens)
    return {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": 0,
        "model": req.model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text,
                },
                "finish_reason": "stop"
            }
        ],
    }

@app.post("/v1/completions")
async def completions(req: ChatRequest):
    # Single-prompt compatibility for get_completion()
    prompt = ""
    for msg in req.messages:
        if msg["role"].lower() == "user":
            prompt += msg["content"] + "\n"
    response_text = generate_response(prompt, max_new_tokens=req.max_tokens)
    return {
        "id": "cmpl-123",
        "object": "text_completion",
        "created": 0,
        "model": req.model,
        "choices": [
            {
                "text": response_text,
                "index": 0,
                "finish_reason": "stop"
            }
        ],
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000)

# run locally with uvicorn main:app --host 0.0.0.0 --port 5000 --reload

