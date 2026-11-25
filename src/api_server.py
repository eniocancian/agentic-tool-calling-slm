import os
import json
from typing import List, Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = os.getenv("BASE_MODEL_ID", "Qwen/Qwen2.5-3B-Instruct")
ADAPTER_DIR = os.getenv("LORA_ADAPTER_DIR", "")

app = FastAPI(title="Agentic Tool-Calling SLM Router", version="1.0.0")

tokenizer = None
model = None


class RouteRequest(BaseModel):
    user_query: str
    tools: List[Dict[str, Any]]


@app.on_event("startup")
def load_model() -> None:
    global tokenizer, model
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16, device_map="auto")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR) if ADAPTER_DIR and os.path.exists(ADAPTER_DIR) else base
    model.eval()


@app.post("/v1/route")
def route_tool_call(req: RouteRequest) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    messages = [
        {"role": "system", "content": f"You are a deterministic function-calling router. Available tools:\n{json.dumps(req.tools, ensure_ascii=False)}"},
        {"role": "user", "content": req.user_query},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=384, do_sample=False, temperature=0.0)

    raw_response = tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    try:
        return {"status": "ok", "decision": json.loads(raw_response)}
    except json.JSONDecodeError:
        return {"status": "parse_error", "raw_output": raw_response}
