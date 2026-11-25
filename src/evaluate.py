import json
import sys
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parents[1] / "data"))
from custom_tools_schema import AgentDecision


def parse_and_validate(raw_text: str) -> tuple[bool, bool, dict | None]:
    """Returns (is_valid_json, is_valid_schema, parsed_dict)."""
    cleaned = raw_text.strip()
    if cleaned.startswith("```json"):
        cleaned = cleaned.removeprefix("```json").removesuffix("```").strip()
    elif cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```").removesuffix("```").strip()
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        return False, False, None

    try:
        AgentDecision.model_validate(data)
        return True, True, data
    except Exception:
        return True, False, data


def run_benchmark(model_id: str, adapter_path: str | None, eval_file: Path, max_samples: int = 200) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    records = [json.loads(line) for line in eval_file.open("r", encoding="utf-8")][:max_samples]
    metrics = {"total": len(records), "json_valid": 0, "schema_valid": 0, "tool_exact_match": 0}

    for rec in tqdm(records, desc=f"Evaluating {adapter_path or model_id}"):
        prompt_msgs = rec["messages"][:2]
        expected = json.loads(rec["messages"][2]["content"])

        prompt_text = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=384, do_sample=False, temperature=0.0)

        generated = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        valid_json, valid_schema, parsed = parse_and_validate(generated)

        if valid_json:
            metrics["json_valid"] += 1
        if valid_schema:
            metrics["schema_valid"] += 1
            pred_tools = [t["name"] for t in parsed.get("tool_calls", [])]
            exp_tools = [t["name"] for t in expected.get("tool_calls", [])]
            if parsed.get("action") == expected.get("action") and pred_tools == exp_tools:
                metrics["tool_exact_match"] += 1

    n = metrics["total"]
    return {
        "model": adapter_path or model_id,
        "samples": n,
        "json_validity_pct": round(100 * metrics["json_valid"] / n, 2),
        "schema_compliance_pct": round(100 * metrics["schema_valid"] / n, 2),
        "tool_selection_acc_pct": round(100 * metrics["tool_exact_match"] / n, 2),
    }


if __name__ == "__main__":
    eval_path = Path(__file__).resolve().parents[1] / "data" / "eval.jsonl"
    if not eval_path.exists():
        raise FileNotFoundError("Run data/prepare_dataset.py first to generate eval.jsonl")
    res = run_benchmark("Qwen/Qwen2.5-3B-Instruct", adapter_path=None, eval_file=eval_path, max_samples=100)
    print(json.dumps(res, indent=2))
