import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

from app.risk_model import calculate_risk
from app.scenario import run_scenarios
from app.action import simulate_actions
from app.llm import generate_advice, generate_chat_reply

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    html_path = os.path.join(os.path.dirname(__file__), "..", "public", "index.html")
    try:
        with open(html_path, "r", encoding="utf-8") as f:
            content = f.read()
        return HTMLResponse(content=content)
    except FileNotFoundError:
        return {"status": "ok"}


@app.post("/analyze")
def analyze(data: dict):
    # 필수 필드 검증
    required = ["income", "expense", "debt", "assets"]
    for field in required:
        if data.get(field) is None:
            raise HTTPException(status_code=422, detail=f"필수 필드 누락: {field}")

    # 1. 기본 리스크 계산
    risk = calculate_risk(data)

    # 2. 시나리오 분석
    scenarios = run_scenarios(data)

    # 3. 행동 시뮬레이션
    action_simulations = simulate_actions(data)

    # 4. AI Agent 심층 분석
    payload = {
        "input": data,
        "risk": risk,
        "scenarios": scenarios,
        "action_simulations": action_simulations,
    }
    advice = generate_advice(payload)

    return {
        "risk": risk,
        "scenarios": scenarios,
        "action_simulations": action_simulations,
        "advice": advice,
    }


@app.post("/chat")
def chat(payload: dict):
    message = payload.get("message", "").strip()
    context = payload.get("context", {})
    if not message:
        raise HTTPException(status_code=422, detail="message is required")
    reply = generate_chat_reply(message, context)
    return {"reply": reply}
