import json
import random
from pathlib import Path
from datasets import load_dataset
from custom_tools_schema import DOMAIN_TOOLS_CATALOG

SEED = 42
TRAIN_SIZE = 3000
EVAL_SIZE = 400
OUTPUT_DIR = Path(__file__).resolve().parent


def build_synthetic_edge_cases() -> list[dict]:
    """Generates high-precision domain examples including no-tool and missing-parameter refusals."""
    tools_json = json.dumps(DOMAIN_TOOLS_CATALOG, ensure_ascii=False)
    samples = [
        {
            "category": "single_tool",
            "messages": [
                {"role": "system", "content": f"You are a deterministic function-calling router. Available tools:\n{tools_json}"},
                {"role": "user", "content": "Check the temperature and pressure of heat exchanger HX-204 over the last 30 minutes."},
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "action": "call_tool",
                        "tool_calls": [
                            {"name": "get_equipment_telemetry", "arguments": {"asset_id": "HX-204", "metric": "temperature", "window_minutes": 30}},
                            {"name": "get_equipment_telemetry", "arguments": {"asset_id": "HX-204", "metric": "pressure", "window_minutes": 30}},
                        ],
                        "message": None,
                    }, ensure_ascii=False),
                },
            ],
        },
        {
            "category": "missing_params",
            "messages": [
                {"role": "system", "content": f"You are a deterministic function-calling router. Available tools:\n{tools_json}"},
                {"role": "user", "content": "Run a VaR risk check on my portfolio."},
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "action": "clarify",
                        "tool_calls": [],
                        "message": "Please provide the list of asset tickers and desired confidence level (0.95 or 0.99).",
                    }, ensure_ascii=False),
                },
            ],
        },
        {
            "category": "no_tool",
            "messages": [
                {"role": "system", "content": f"You are a deterministic function-calling router. Available tools:\n{tools_json}"},
                {"role": "user", "content": "What is the physical definition of Reynolds number?"},
                {
                    "role": "assistant",
                    "content": json.dumps({
                        "action": "no_tool",
                        "tool_calls": [],
                        "message": "No external tool is required for theoretical definitions.",
                    }, ensure_ascii=False),
                },
            ],
        },
    ]
    return samples * 100


def format_xlam_record(row: dict) -> dict | None:
    try:
        tools = json.loads(row["tools"]) if isinstance(row["tools"], str) else row["tools"]
        answers = json.loads(row["answers"]) if isinstance(row["answers"], str) else row["answers"]
        if not answers:
            return None
        category = "parallel_tool" if len(answers) > 1 else "single_tool"
        assistant_payload = json.dumps({
            "action": "call_tool",
            "tool_calls": [{"name": a["name"], "arguments": a["arguments"]} for a in answers],
            "message": None,
        }, ensure_ascii=False)
        return {
            "category": category,
            "messages": [
                {"role": "system", "content": f"You are a deterministic function-calling router. Available tools:\n{json.dumps(tools, ensure_ascii=False)}"},
                {"role": "user", "content": row["query"]},
                {"role": "assistant", "content": assistant_payload},
            ],
        }
    except Exception:
        return None


def main() -> None:
    random.seed(SEED)
    print("Loading Salesforce/xlam-function-calling-60k stream...")
    ds = load_dataset("Salesforce/xlam-function-calling-60k", split="train", streaming=True)

    curated = []
    for row in ds:
        formatted = format_xlam_record(row)
        if formatted:
            curated.append(formatted)
        if len(curated) >= (TRAIN_SIZE + EVAL_SIZE):
            break

    curated.extend(build_synthetic_edge_cases())
    random.shuffle(curated)

    train_data = curated[:TRAIN_SIZE]
    eval_data = curated[TRAIN_SIZE : TRAIN_SIZE + EVAL_SIZE]

    for split_name, records in [("train", train_data), ("eval", eval_data)]:
        out_path = OUTPUT_DIR / f"{split_name}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for item in records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved {len(records)} records to {out_path}")


if __name__ == "__main__":
    main()
