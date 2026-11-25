# Benchmark Report: Base SLM vs. QLoRA Fine-Tuned SLM

| Model Variant | JSON Syntax Validity (%) | Pydantic Schema Compliance (%) | Tool Selection Accuracy (%) |
| :--- | :---: | :---: | :---: |
| `Qwen2.5-3B-Instruct` (Zero-Shot Base) | _Pending_ | _Pending_ | _Pending_ |
| `Qwen2.5-3B-Instruct` + QLoRA (`r=16, alpha=32`) | _Pending_ | _Pending_ | _Pending_ |

## 3. Live Inference & Residual Error Analysis

Qualitative stress-testing against unseen domain prompts revealed strong generalization alongside one residual failure mode:
- **Parallel Tool Decomposition (Pass):** Correctly decomposed `"temperature and vibration of heat exchanger HX-402 over the last 45 minutes"` into two parallel `get_equipment_telemetry` calls with exact argument extraction.
- **Out-of-Scope Routing (Pass):** Correctly routed theoretical thermodynamic queries (`LMTD vs. NTU`) to `"action": "no_tool"` without hallucinating tool calls.
- **Missing Parameter Hallucination (Residual Error):** Due to class imbalance in the training mix (~91% direct tool invocations from `glaive-function-calling-v2` vs. ~3% `clarify` examples), under-specified prompts (`"Calculate the VaR exposure for my stock portfolio"`) triggered default parameter fabrication (`["AAPL", "GOOGL", "MSFT"]`, `0.95`) instead of `"action": "clarify"`. Mitigating this behavior is the primary target for a subsequent **DPO (Direct Preference Optimization)** alignment pass.
