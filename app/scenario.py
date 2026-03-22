from app.risk_model import calculate_risk

# 각 시나리오의 대전제(테마)와 방향성 정의
SCENARIOS = [
    {
        "name": "소득 -10%",
        "theme": "소득 충격 시나리오",
        "premise": "직장 변경, 구조조정 등으로 연 소득이 10% 줄어든다면?",
        "direction": "negative",
        "icon": "📉",
    },
    {
        "name": "경기 침체 (자산 -20%)",
        "theme": "자산 가치 하락 시나리오",
        "premise": "주식·부동산 등 보유 자산의 시장 가치가 20% 하락한다면?",
        "direction": "negative",
        "icon": "🏦",
    },
    {
        "name": "지출 -15%",
        "theme": "절약 실천 시나리오",
        "premise": "소비 습관 개선으로 연간 지출을 15% 줄인다면?",
        "direction": "positive",
        "icon": "✂️",
    },
    {
        "name": "부채 20% 상환",
        "theme": "부채 탈출 시나리오",
        "premise": "적극적인 상환 전략으로 총 부채를 20% 줄인다면?",
        "direction": "positive",
        "icon": "💳",
    },
    {
        "name": "복합 위기 (소득↓ 지출↑)",
        "theme": "최악의 복합 위기 시나리오",
        "premise": "경기 악화로 소득은 20% 줄고 생활비는 10% 오른다면?",
        "direction": "crisis",
        "icon": "⚠️",
    },
]


def _apply_scenario(data: dict, scenario: dict) -> dict:
    d = data.copy()
    name = scenario["name"]

    if name == "소득 -10%":
        d["income"] = d["income"] * 0.9
    elif name == "경기 침체 (자산 -20%)":
        d["assets"] = d.get("assets", 0) * 0.8
    elif name == "지출 -15%":
        d["expense"] = d["expense"] * 0.85
    elif name == "부채 20% 상환":
        d["debt"] = d["debt"] * 0.8
    elif name == "복합 위기 (소득↓ 지출↑)":
        d["income"] = d["income"] * 0.8
        d["expense"] = d["expense"] * 1.1

    return d


def run_scenarios(data: dict) -> list:
    scenarios = []
    for s in SCENARIOS:
        modified = _apply_scenario(data, s)
        result = calculate_risk(modified)
        scenarios.append({
            "name": s["name"],
            "theme": s["theme"],
            "premise": s["premise"],
            "direction": s["direction"],
            "icon": s["icon"],
            "result": result,
        })
    return scenarios
