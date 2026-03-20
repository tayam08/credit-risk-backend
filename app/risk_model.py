DELINQUENCY_SCORE = {"없음": 0, "1회": 5, "다수": 10}


def calculate_metrics(data):
    income = data["income"]
    expense = data["expense"]
    debt = data["debt"]
    assets = data.get("assets", 0)

    income = max(income, 1)

    # 부채비율: 자산 대비 부채 비중 (%)
    # 예) 부채 15000 / 자산 20000 = 75% → "자산의 75%가 부채"
    debt_to_asset_pct = round(debt / max(assets, 1) * 100, 1)

    # 자금 유지기간 재설계
    annual_surplus = income - expense  # 연간 잉여(+) 또는 적자(-)

    if annual_surplus > 0:
        if debt > 0:
            # 흑자: 현재 저축 속도로 부채 전액 상환까지 걸리는 기간
            survival_years = round(debt / annual_surplus, 1)
            survival_mode = "debt_payoff"   # "부채 상환까지 N년"
        else:
            survival_years = 0
            survival_mode = "stable"        # 부채 없고 흑자 → 안정
    else:
        # 적자: 순자산(자산-부채)이 소진되는 기간
        net_worth = max(assets - debt, 0)
        deficit = abs(annual_surplus)
        survival_years = round(net_worth / deficit, 1) if deficit > 0 else 0
        survival_mode = "depletion"         # "자산 소진까지 N년"

    return {
        "debt_to_asset_pct": debt_to_asset_pct,
        "survival_years": survival_years,
        "survival_mode": survival_mode,
    }


def calculate_risk(data):
    income = data["income"]
    expense = data["expense"]
    debt = data["debt"]

    # 선택 필드 - 없으면 중립값 사용
    credit_score = data.get("credit_score") or 650
    delinquency = data.get("delinquency", "없음") or "없음"

    income = max(income, 1)

    debt_ratio = debt / income       # 내부 리스크 계산용 (부채/수입)
    expense_ratio = expense / income

    # delinquency: 문자열("없음"/"1회"/"다수") 또는 bool 모두 지원
    if isinstance(delinquency, bool):
        delinquency_score = 10 if delinquency else 0
    else:
        delinquency_score = DELINQUENCY_SCORE.get(str(delinquency), 0)

    # 각 요소를 0~1로 정규화한 뒤 가중치 적용 (합계 최대 100점)
    # 부채/수입 비율: 수입의 5배 이상 = 최고 위험 (40점)
    debt_score = min(debt_ratio / 5.0, 1.0) * 40

    # 지출/수입 비율: 수입의 1.5배 이상 = 최고 위험 (30점)
    expense_score = min(expense_ratio / 1.5, 1.0) * 30

    # 신용점수: 300~700 범위 정규화, 낮을수록 위험 (20점)
    clamped_credit = max(min(credit_score, 700), 300)
    credit_score_risk = (700 - clamped_credit) / 400.0 * 20

    score = debt_score + expense_score + credit_score_risk + delinquency_score

    probability = min(round(score, 1), 100)

    if probability < 30:
        level = "낮음"
    elif probability < 60:
        level = "보통"
    else:
        level = "높음"

    metrics = calculate_metrics(data)

    return {
        "risk_score": round(score, 2),
        "probability": probability,
        "level": level,
        "expense_ratio": round(expense_ratio, 2),
        "debt_to_asset_pct": metrics["debt_to_asset_pct"],
        "survival_years": metrics["survival_years"],
        "survival_mode": metrics["survival_mode"],
    }
