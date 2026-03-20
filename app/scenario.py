from app.risk_model import calculate_risk

def run_scenarios(data):
    scenarios = []

    # 📉 소득 감소
    s1 = data.copy()
    s1["income"] *= 0.9
    scenarios.append({
        "name": "소득 -10%",
        "result": calculate_risk(s1)
    })

    # 📉 경기 침체 (자산 감소)
    s2 = data.copy()
    s2["assets"] = s2.get("assets", 0) * 0.8
    scenarios.append({
        "name": "경기 침체 (자산 -20%)",
        "result": calculate_risk(s2)
    })

    # 📈 지출 감소
    s3 = data.copy()
    s3["expense"] *= 0.85
    scenarios.append({
        "name": "지출 -15%",
        "result": calculate_risk(s3)
    })

    # 📈 부채 상환
    s4 = data.copy()
    s4["debt"] *= 0.8
    scenarios.append({
        "name": "부채 20% 상환",
        "result": calculate_risk(s4)
    })

    # 📉 극단 상황
    s5 = data.copy()
    s5["income"] *= 0.8
    s5["expense"] *= 1.1
    scenarios.append({
        "name": "위기 상황 (소득↓ 지출↑)",
        "result": calculate_risk(s5)
    })

    return scenarios