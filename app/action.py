from app.risk_model import calculate_risk

def simulate_actions(data):
    result = {}

    # 📉 지출 감소
    d1 = data.copy()
    d1["expense"] *= 0.85
    result["expense_reduction"] = calculate_risk(d1)

    # 📉 부채 상환
    d2 = data.copy()
    d2["debt"] *= 0.8
    result["debt_reduction"] = calculate_risk(d2)

    # 📈 소득 증가
    d3 = data.copy()
    d3["income"] *= 1.1
    result["income_increase"] = calculate_risk(d3)

    return result