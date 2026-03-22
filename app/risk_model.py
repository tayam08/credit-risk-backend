import math

DELINQUENCY_SCORE = {"없음": 0, "1회": 5, "다수": 10}

# ─────────────────────────────────────────
# 나이대별 리스크 점수 분포 (합성 통계, 정규분포 가정)
# 한국 연령대별 일반적 재무 부담 수준 반영
# ─────────────────────────────────────────
AGE_GROUP_STATS = {
    "10대": {"mean": 10, "std": 6},     # 학생 중심, 부채 거의 없음
    "20대": {"mean": 26, "std": 14},    # 사회초년생, 학자금/소비성 부채 시작
    "30대": {"mean": 42, "std": 17},    # 주택담보대출, 가정 형성기
    "40대": {"mean": 38, "std": 16},    # 고소득 but 최대 부채 구간
    "50대": {"mean": 31, "std": 14},    # 부채 상환기, 노후 준비
    "60대 이상": {"mean": 22, "std": 12},  # 은퇴 후, 소득 감소
}


def get_age_group(age: int) -> str:
    if age < 20:
        return "10대"
    elif age < 30:
        return "20대"
    elif age < 40:
        return "30대"
    elif age < 50:
        return "40대"
    elif age < 60:
        return "50대"
    else:
        return "60대 이상"


def _normal_cdf(x: float, mean: float, std: float) -> float:
    """정규 분포 누적 분포 함수 (math.erf 이용, scipy 불필요)"""
    return 0.5 * (1.0 + math.erf((x - mean) / (std * math.sqrt(2))))


def get_age_percentile(age: int, risk_score: float) -> dict:
    """
    동 나이대 내에서의 리스크 점수 백분위 반환.
    percentile = 동 나이대 중 이 점수보다 낮은 리스크를 가진 사람의 비율(%)
    예) percentile=75 → 75%보다 리스크가 높음 → 상위 25%
    """
    age_group = get_age_group(age)
    stats = AGE_GROUP_STATS[age_group]
    percentile = _normal_cdf(risk_score, stats["mean"], stats["std"]) * 100
    percentile = round(min(max(percentile, 0.5), 99.5), 1)

    top_pct = round(100 - percentile, 1)

    if percentile >= 50:
        label = f"{age_group} 중 상위 {top_pct}%"
        risk_position = "high_relative"  # 나이대 평균보다 위험
    else:
        label = f"{age_group} 중 하위 {round(percentile, 1)}%"
        risk_position = "low_relative"   # 나이대 평균보다 안전

    return {
        "age_group": age_group,
        "percentile": percentile,       # 0~100, 높을수록 동 나이대 대비 위험
        "top_percent": top_pct,         # 상위 X% (작을수록 위험)
        "label": label,
        "risk_position": risk_position,
    }


def calculate_metrics(data):
    income = data["income"]
    expense = data["expense"]
    debt = data["debt"]
    assets = data.get("assets", 0)

    income = max(income, 1)

    # 부채비율: 자산 대비 부채 비중 (%)
    debt_to_asset_pct = round(debt / max(assets, 1) * 100, 1)

    # 자금 유지기간 재설계
    annual_surplus = income - expense

    if annual_surplus > 0:
        if debt > 0:
            survival_years = round(debt / annual_surplus, 1)
            survival_mode = "debt_payoff"
        else:
            survival_years = 0
            survival_mode = "stable"
    else:
        net_worth = max(assets - debt, 0)
        deficit = abs(annual_surplus)
        survival_years = round(net_worth / deficit, 1) if deficit > 0 else 0
        survival_mode = "depletion"

    return {
        "debt_to_asset_pct": debt_to_asset_pct,
        "survival_years": survival_years,
        "survival_mode": survival_mode,
    }


def calculate_risk(data):
    income = data["income"]
    expense = data["expense"]
    debt = data["debt"]

    credit_score = data.get("credit_score") or 650
    delinquency = data.get("delinquency", "없음") or "없음"
    age = data.get("age")

    income = max(income, 1)

    debt_ratio = debt / income
    expense_ratio = expense / income

    if isinstance(delinquency, bool):
        delinquency_score = 10 if delinquency else 0
    else:
        delinquency_score = DELINQUENCY_SCORE.get(str(delinquency), 0)

    debt_score = min(debt_ratio / 5.0, 1.0) * 40
    expense_score = min(expense_ratio / 1.5, 1.0) * 30
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

    result = {
        "risk_score": round(score, 2),
        "probability": probability,
        "level": level,
        "expense_ratio": round(expense_ratio, 2),
        "debt_ratio": round(debt_ratio, 2),
        "debt_to_asset_pct": metrics["debt_to_asset_pct"],
        "survival_years": metrics["survival_years"],
        "survival_mode": metrics["survival_mode"],
    }

    if age:
        result["age_percentile"] = get_age_percentile(age, score)

    return result
