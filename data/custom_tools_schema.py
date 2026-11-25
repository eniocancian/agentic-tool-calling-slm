from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    name: str = Field(..., description="Exact name of the target function to invoke")
    arguments: dict = Field(default_factory=dict, description="Key-value payload matching tool schema")


class AgentDecision(BaseModel):
    action: Literal["call_tool", "no_tool", "clarify"] = Field(
        ..., description="Routing decision taken by the SLM"
    )
    tool_calls: List[ToolCall] = Field(
        default_factory=list, description="List of zero, one, or multiple parallel tool calls"
    )
    message: Optional[str] = Field(
        default=None, description="User-facing clarification or direct response when no tool is invoked"
    )


DOMAIN_TOOLS_CATALOG = [
    {
        "type": "function",
        "function": {
            "name": "get_equipment_telemetry",
            "description": "Fetch real-time sensor metrics (temperature, pressure, vibration) for an industrial asset.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string", "description": "Unique asset tag, e.g., HX-104"},
                    "metric": {"type": "string", "enum": ["temperature", "pressure", "vibration", "all"]},
                    "window_minutes": {"type": "integer", "default": 60},
                },
                "required": ["asset_id", "metric"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "query_portfolio_exposure",
            "description": "Calculate sector exposure and VaR for a given portfolio ticker list.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tickers": {"type": "array", "items": {"type": "string"}},
                    "confidence_level": {"type": "number", "enum": [0.95, 0.99]},
                },
                "required": ["tickers", "confidence_level"],
            },
        },
    },
]
