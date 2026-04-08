import uuid
from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from inference import TASKS, FALLBACK_RESULTS, call_llm, step

app = FastAPI(title="RegretZero", version="1.0.0")


class InferenceRequest(BaseModel):
    session_id: Optional[str] = None


@app.get("/")
def root():
    return {"status": "ok", "message": "RegretZero is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/run")
def run_inference(request: InferenceRequest = None):
    session_id = (request.session_id if request else None) or str(uuid.uuid4())
    all_results = []

    for task in TASKS:
        try:
            llm_result = call_llm(task["decision"], task["stakes"], task["timeframe"])
            verdict = "LLM"

            if llm_result is None:
                llm_result = FALLBACK_RESULTS[task["id"]]
                verdict = "FALLBACK"

            step_result = step(task, llm_result)
            all_results.append({
                "task_id": task["id"],
                "session_id": session_id,
                "suggestion": llm_result.get("suggestion", ""),
                "regret_risk": llm_result.get("regret_risk", 0.5),
                "confidence": llm_result.get("confidence", 0.5),
                "verdict": verdict,
                "reward": step_result["reward"]
            })

        except Exception as e:
            fallback = FALLBACK_RESULTS[task["id"]]
            step_result = step(task, fallback)
            all_results.append({
                "task_id": task["id"],
                "session_id": session_id,
                "suggestion": fallback["suggestion"],
                "regret_risk": fallback["regret_risk"],
                "confidence": fallback["confidence"],
                "verdict": "FALLBACK",
                "reward": step_result["reward"],
                "error": str(e)
            })

    return {"session_id": session_id, "results": all_results}


@app.get("/tasks")
def get_tasks():
    return {"tasks": TASKS}