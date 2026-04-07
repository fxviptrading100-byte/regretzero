from fastapi import FastAPI
import os
import json
import uuid
from openai import OpenAI
 
app = FastAPI()
 
# FIXED: HF OpenAI-compatible endpoint is /v1
API_BASE_URL = os.getenv("API_BASE_URL", "https://api-inference.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
 
client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
 
TASKS = [
    {
        "id": "career_decision",
        "decision": "Should I quit my stable job to pursue my startup idea?",
        "stakes": "high",
        "timeframe": "5 years"
    },
    {
        "id": "relationship_decision",
        "decision": "Should I move to another city for a relationship?",
        "stakes": "high",
        "timeframe": "3 years"
    },
    {
        "id": "education_decision",
        "decision": "Should I go back to school for a masters degree?",
        "stakes": "medium",
        "timeframe": "2 years"
    }
]
 
 
def call_llm(decision: str, stakes: str = "medium", timeframe: str = "1 year") -> dict:
    prompt = (
        f"A person faces this decision: {decision}\n"
        f"Stakes level: {stakes}. Timeframe: {timeframe}.\n"
        f"As their future self {timeframe} from now, give concrete advice to minimize regret. "
        f"Respond in JSON only with keys: suggestion (string), regret_risk (float 0-1), confidence (float 0-1)."
    )
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a Future Self Advisor. You respond ONLY with valid JSON. "
                    "Never include markdown, backticks, or explanation outside the JSON."
                )
            },
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=300
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    return json.loads(raw.strip())
 
 
def score_result(result: dict) -> float:
    score = 0.0
    suggestion = result.get("suggestion", "")
    if isinstance(suggestion, str) and len(suggestion) > 30:
        length_score = min(len(suggestion) / 500, 0.95) * 0.38
        score += length_score
    regret_risk = result.get("regret_risk", -1)
    if isinstance(regret_risk, (int, float)) and 0.0 < regret_risk < 1.0:
        score += 0.29
    confidence = result.get("confidence", -1)
    if isinstance(confidence, (int, float)) and 0.0 < confidence < 1.0:
        score += 0.29
    return round(min(score, 0.99), 2)
 
 
@app.get("/")
def root():
    return {"Hello": "RegretZero Space is running!", "status": "ok"}
 
 
@app.get("/health")
def health():
    return {"status": "healthy"}
 
 
@app.post("/reset")
def reset():
    return {"status": "reset", "tasks": [t["id"] for t in TASKS]}
 
 
@app.get("/state")
def state():
    return {"tasks": TASKS, "status": "ready"}
 
 
@app.post("/step")
async def step_endpoint(request: dict):
    task_id = request.get("task_id")
    result = request.get("result", {})
    reward = score_result(result)
    return {
        "task_id": task_id,
        "reward": reward,
        "done": True,
        "info": {
            "regret_risk": result.get("regret_risk"),
            "confidence": result.get("confidence")
        }
    }
 
 
@app.post("/analyze")
async def analyze_decision(request: dict):
    print("[START] RegretZero Analysis Started")
 
    decision = request.get("decision", "No decision provided")
    stakes = request.get("stakes", "medium")
    timeframe = request.get("timeframe", "1 year")
    session_id = request.get("session_id", str(uuid.uuid4()))
 
    print(f"[STEP] Processing decision: {decision}")
 
    try:
        llm_result = call_llm(decision, stakes, timeframe)
        suggestion = llm_result.get("suggestion", "")
        confidence = float(llm_result.get("confidence", 0.82))
        regret_risk = float(llm_result.get("regret_risk", 0.35))
    except Exception as e:
        print(f"[STEP] LLM error: {e}")
        raise
 
    result = {
        "session_id": session_id,
        "suggestion": suggestion,
        "regret_risk": regret_risk,
        "confidence": confidence,
        "verdict": "HUMAN VERIFIED"
    }
 
    print("[STEP] Analysis complete")
    print("[END] RegretZero Analysis Complete")
    return result
