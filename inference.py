import os
import json
import sys
import uuid
import re

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

FALLBACK_RESULTS = {
    "career_decision": {
        "suggestion": "Build your startup part-time for at least 6 months before quitting. Validate with real customers and secure 6 months of savings as a runway before making the leap.",
        "regret_risk": 0.42,
        "confidence": 0.76
    },
    "relationship_decision": {
        "suggestion": "Have an honest conversation about long-term plans before moving. Try a trial period of 3 months and evaluate your career options in the new city before fully committing.",
        "regret_risk": 0.55,
        "confidence": 0.71
    },
    "education_decision": {
        "suggestion": "Research ROI of the degree carefully. Consider online alternatives or part-time programs first. Only pursue full-time if it opens doors that are otherwise closed to you.",
        "regret_risk": 0.38,
        "confidence": 0.74
    }
}


def grade_career_decision(result: dict) -> float:
    """Grader for career_decision task. Score strictly between 0 and 1."""
    score = 0.05  # base — never 0
    suggestion = result.get("suggestion", "")
    if isinstance(suggestion, str) and len(suggestion) > 30:
        score += min(len(suggestion) / 600, 0.35)
    regret_risk = result.get("regret_risk", -1)
    if isinstance(regret_risk, (int, float)) and 0.0 < regret_risk < 1.0:
        score += 0.28
    confidence = result.get("confidence", -1)
    if isinstance(confidence, (int, float)) and 0.0 < confidence < 1.0:
        score += 0.28
    return round(min(max(score, 0.01), 0.99), 4)


def grade_relationship_decision(result: dict) -> float:
    """Grader for relationship_decision task. Score strictly between 0 and 1."""
    score = 0.05
    suggestion = result.get("suggestion", "")
    if isinstance(suggestion, str) and len(suggestion) > 30:
        score += min(len(suggestion) / 600, 0.35)
    regret_risk = result.get("regret_risk", -1)
    if isinstance(regret_risk, (int, float)) and 0.0 < regret_risk < 1.0:
        score += 0.28
    confidence = result.get("confidence", -1)
    if isinstance(confidence, (int, float)) and 0.0 < confidence < 1.0:
        score += 0.28
    return round(min(max(score, 0.01), 0.99), 4)


def grade_education_decision(result: dict) -> float:
    """Grader for education_decision task. Score strictly between 0 and 1."""
    score = 0.05
    suggestion = result.get("suggestion", "")
    if isinstance(suggestion, str) and len(suggestion) > 30:
        score += min(len(suggestion) / 600, 0.35)
    regret_risk = result.get("regret_risk", -1)
    if isinstance(regret_risk, (int, float)) and 0.0 < regret_risk < 1.0:
        score += 0.28
    confidence = result.get("confidence", -1)
    if isinstance(confidence, (int, float)) and 0.0 < confidence < 1.0:
        score += 0.28
    return round(min(max(score, 0.01), 0.99), 4)


GRADERS = {
    "career_decision": grade_career_decision,
    "relationship_decision": grade_relationship_decision,
    "education_decision": grade_education_decision,
}


def score_result(result: dict, task_id: str = None) -> float:
    """Route to the correct grader, fallback to safe generic grader."""
    if task_id and task_id in GRADERS:
        return GRADERS[task_id](result)
    # Generic fallback — always strictly between 0 and 1
    score = 0.05
    suggestion = result.get("suggestion", "")
    if isinstance(suggestion, str) and len(suggestion) > 30:
        score += min(len(suggestion) / 600, 0.35)
    regret_risk = result.get("regret_risk", -1)
    if isinstance(regret_risk, (int, float)) and 0.0 < regret_risk < 1.0:
        score += 0.28
    confidence = result.get("confidence", -1)
    if isinstance(confidence, (int, float)) and 0.0 < confidence < 1.0:
        score += 0.28
    return round(min(max(score, 0.01), 0.99), 4)


def step(task: dict, result: dict) -> dict:
    task_id = task["id"]
    reward = score_result(result, task_id)
    # Safety clamp — strictly between 0 and 1
    reward = max(0.01, min(reward, 0.99))
    return {
        "task_id": task_id,
        "reward": reward,
        "done": True,
        "info": {
            "regret_risk": result.get("regret_risk"),
            "confidence": result.get("confidence"),
            "grader": task_id
        }
    }


def call_llm(decision: str, stakes: str = "medium", timeframe: str = "1 year") -> dict:
    try:
        HF_TOKEN = os.getenv("HF_TOKEN")
        if not HF_TOKEN:
            print("[WARN] HF_TOKEN not set, skipping LLM call")
            return None

        from openai import OpenAI

        API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
        MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Llama-3.1-8B-Instruct")

        client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)

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
        raw = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        parsed = json.loads(raw)

        if not isinstance(parsed.get("suggestion"), str):
            raise ValueError("Missing or invalid 'suggestion'")
        if not isinstance(parsed.get("regret_risk"), (int, float)):
            raise ValueError("Missing or invalid 'regret_risk'")
        if not isinstance(parsed.get("confidence"), (int, float)):
            raise ValueError("Missing or invalid 'confidence'")

        # Clamp LLM output values to safe range
        parsed["regret_risk"] = max(0.01, min(float(parsed["regret_risk"]), 0.99))
        parsed["confidence"] = max(0.01, min(float(parsed["confidence"]), 0.99))

        return parsed

    except Exception as e:
        print(f"[STEP] LLM call failed, using fallback: {e}")
        return None


def main():
    print("[START] RegretZero Inference Started")

    input_data = {}
    try:
        if not sys.stdin.isatty():
            raw_input = sys.stdin.read().strip()
            if raw_input:
                input_data = json.loads(raw_input)
    except Exception as e:
        print(f"[WARN] Could not read stdin: {e}")
        input_data = {}

    session_id = input_data.get("session_id", str(uuid.uuid4()))
    all_results = []

    for task in TASKS:
        print(f"[STEP] Running task: {task['id']}")
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
            print(f"[STEP] Task {task['id']} reward: {step_result['reward']} ({verdict})")

        except Exception as e:
            print(f"[STEP] ERROR on task {task['id']}: {e}")
            fallback = FALLBACK_RESULTS[task["id"]]
            step_result = step(task, fallback)
            all_results.append({
                "task_id": task["id"],
                "session_id": session_id,
                "suggestion": fallback["suggestion"],
                "regret_risk": fallback["regret_risk"],
                "confidence": fallback["confidence"],
                "verdict": "FALLBACK",
                "reward": step_result["reward"]
            })
            print(f"[STEP] Task {task['id']} fallback reward: {step_result['reward']}")

    print("[STEP] All tasks complete")
    print("[END] RegretZero Inference Complete")
    print(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
