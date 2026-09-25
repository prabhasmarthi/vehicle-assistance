import os
import uvicorn
import requests
import json
from fastapi import FastAPI
from langserve import add_routes
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from pydantic import BaseModel, Field
from langchain_core.runnables import RunnableLambda

# --- 1. Define Vehicle Tools ---
@tool
def lookup_warning_light(indicator_name: str) -> str:
    """Look up vehicle dashboard warning light meanings and urgency levels."""
    database = {
        "check engine": "Severity: HIGH. Indicates engine emission or sensor malfunction. Drive carefully to nearest service center.",
        "battery": "Severity: CRITICAL. Indicates charging system failure or alternator fault. Pull over safely.",
        "oil pressure": "Severity: CRITICAL. Low engine oil pressure. Stop engine immediately to prevent engine damage.",
        "tpms": "Severity: MEDIUM. Tire Pressure Monitoring System indicates low tire pressure."
    }
    return database.get(indicator_name.lower().strip(), "Unknown warning light code. Please consult vehicle manual or visit a certified technician.")

@tool
def get_roadside_assistance(location: str) -> str:
    """Find nearby emergency towing, battery jump-start, and roadside assistance services."""
    services = {
        "mumbai": {"provider": "FastTrack Towing Mumbai", "eta_minutes": 15, "contact": "+91-1800-111-222", "status": "Available 24/7"},
        "delhi": {"provider": "Capital Rescue Towing", "eta_minutes": 20, "contact": "+91-1800-333-444", "status": "Available 24/7"},
        "bangalore": {"provider": "Bangalore Breakdown Assist", "eta_minutes": 12, "contact": "+91-1800-555-666", "status": "Available 24/7"}
    }
    key = location.lower().strip()
    if key in services:
        return json.dumps(services[key])
    return json.dumps({"provider": "National Highway Assist", "eta_minutes": 30, "contact": "+91-1800-999-000", "status": "Available 24/7"})

tools = [lookup_warning_light, get_roadside_assistance]

# --- 2. Initialize Model & Guardrailed Agent ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

llm_flash = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=GEMINI_API_KEY,
    temperature=0
)

agent = create_agent(
    model=llm_flash,
    tools=tools,
    system_prompt=(
        "You are a specialized agent restricted ONLY to vehicle diagnostics, maintenance, dashboard warning lights, and roadside assistance. "
        "For any other roles, topics, questions, or general knowledge outside of vehicle assistance, "
        "you must say exactly: 'I am not authorized to answer questions outside of vehicle assistance.'"
    )
)

class AgentInput(BaseModel):
    input: str = Field(description="Your message to the vehicle assistance agent")

def format_for_agent(x) -> dict:
    user_input = x["input"] if isinstance(x, dict) else x.input
    return {"messages": [("user", user_input)]}

def extract_text_response(agent_output: dict) -> str:
    if not isinstance(agent_output, dict):
        return str(agent_output)

    messages = agent_output.get("messages")
    if messages is None:
        for value in agent_output.values():
            if isinstance(value, dict) and "messages" in value:
                messages = value["messages"]
                break

    if messages:
        last = messages[-1]
        return getattr(last, "content", str(last))

    return str(agent_output)

formatted_agent_chain = (
    RunnableLambda(format_for_agent)
    | agent
    | RunnableLambda(extract_text_response)
).with_types(input_type=AgentInput, output_type=str)

# --- 3. FastAPI App ---
app = FastAPI(title="Real-Time Vehicle Assistance Agent")
add_routes(app, formatted_agent_chain, path="/agent", playground_type="default")

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
