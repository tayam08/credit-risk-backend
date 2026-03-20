"""
AI 기반 개인 신용 리스크 분석
- 성장률 / Advanced 데이터가 있을 때 Python에서 추가 분석 수행
- 모든 분석 결과를 LLM에 전달해 종합 조언 생성
"""

import os
import json
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL = "google/gemma-3n-e4b-it:free"


# ─────────────────────────────────────────
# 추가 분석 모듈 (Python 계산)
# ─────────────────────────────────────────

def analyze_growth(financial: dict, years: int = 5) -> dict:
    """소득/지출 증가율 + 투자수익률 반영 미래 재무 예측"""
    income = financial.get("income", 0)
    expense = financial.get("expense", 0)
    assets = financial.get("assets", 0)
    income_growth = financial.get("income_growth_rate") or 0
    expense_growth = financial.get("expense_growth_rate") or 0
    inv_return = financial.get("investment_return_rate") or 0

    projections = []
    for y in range(1, years + 1):
        proj_income = income * ((1 + income_growth / 100) ** y)
        proj_expense = expense * ((1 + expense_growth / 100) ** y)
        savings = max(proj_income - proj_expense, 0) * y
        proj_assets = assets * ((1 + inv_return / 100) ** y) + savings
        projections.append({
            "year": y,
            "income": round(proj_income),
            "expense": round(proj_expense),
            "assets": round(proj_assets),
            "net_worth": round(proj_assets - financial.get("debt", 0)),
        })

    final = projections[-1]
    trend = "개선" if final["net_worth"] > projections[0]["net_worth"] else "악화"
    return {
        "5년후_예상순자산": final["net_worth"],
        "5년후_예상소득": final["income"],
        "5년후_예상지출": final["expense"],
        "재무추세": trend,
    }


def analyze_retirement(financial: dict) -> dict:
    """은퇴 준비도 분석 (Advanced 모드)"""
    retirement_age = financial.get("retirement_age") or 65
    assets = financial.get("assets", 0)
    debt = financial.get("debt", 0)
    expense = financial.get("expense", 0)
    inv_return = financial.get("investment_return_rate") or 5
    dependents = financial.get("dependents", "없음")
    asset_sale_cost = financial.get("asset_sale_cost") or 0

    years_to_retire = max(retirement_age - 35, 1)  # 현재 나이 35세 가정
    net_assets = max(assets - debt - asset_sale_cost, 0)
    projected = round(net_assets * ((1 + inv_return / 100) ** years_to_retire))

    retirement_expense = round(expense * (1.2 if dependents == "있음" else 1.0))
    sustainable = round(projected / retirement_expense, 1) if retirement_expense > 0 else 999

    if sustainable >= 25:
        readiness = "충분 (25년 이상 유지 가능)"
    elif sustainable >= 15:
        readiness = "보통 (15~25년 유지 가능)"
    else:
        readiness = "부족 (15년 미만)"

    return {
        "은퇴까지_남은기간": f"{years_to_retire}년",
        "은퇴시_예상자산": projected,
        "연간_은퇴생활비": retirement_expense,
        "자금_지속가능기간": f"{sustainable}년",
        "은퇴준비상태": readiness,
        "부양가족여부": dependents,
    }


# ─────────────────────────────────────────
# LLM 호출
# ─────────────────────────────────────────

def _call_llm(prompt: str) -> str:
    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": MODEL,
            "temperature": 0.2,
            "messages": [
                {"role": "user", "content": prompt},
            ],
        },
        timeout=60,
    )
    data = resp.json()
    if "choices" not in data:
        print(f"[LLM API ERROR] {data}")
        raise KeyError(f"'choices' missing: {data}")
    return data["choices"][0]["message"]["content"]


def _parse_json(text: str) -> dict:
    text = text.strip()

    # <think>...</think> 태그 제거 (일부 모델)
    if "<think>" in text:
        end = text.find("</think>")
        if end != -1:
            text = text[end + len("</think>"):].strip()

    # 코드블록 추출
    for marker in ["```json", "```"]:
        if marker in text:
            text = text.split(marker)[1].split("```")[0].strip()
            break

    # JSON 객체 범위 추출 (앞뒤 텍스트 무시)
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        text = text[start:end]

    return json.loads(text)


# ─────────────────────────────────────────
# 메인 분석 함수
# ─────────────────────────────────────────

def generate_advice(payload: dict) -> dict:
    financial = payload.get("input", payload)
    risk = payload.get("risk", {})
    scenarios = payload.get("scenarios", [])
    action_sims = payload.get("action_simulations", {})

    # 추가 분석 수행 (데이터가 있을 때만)
    extra_sections = []

    has_growth = bool(financial.get("income_growth_rate") or financial.get("expense_growth_rate"))
    has_advanced = bool(financial.get("retirement_age"))

    if has_growth or financial.get("investment_return_rate"):
        growth = analyze_growth(financial)
        extra_sections.append(f"[미래 재무 예측 (5년)]\n{json.dumps(growth, ensure_ascii=False, indent=2)}")

    if has_advanced:
        retirement = analyze_retirement(financial)
        extra_sections.append(f"[은퇴 준비도 분석]\n{json.dumps(retirement, ensure_ascii=False, indent=2)}")

    # 파생 수치 계산 (프롬프트에서 구체적 금액 언급용)
    income = financial.get("income", 0)
    expense = financial.get("expense", 0)
    debt = financial.get("debt", 0)
    assets = financial.get("assets", 0)
    credit_score = financial.get("credit_score")
    delinquency = financial.get("delinquency", "없음")

    monthly_surplus = round((income - expense) / 12, 1)
    expense_cut_amount = round(expense * 0.15)       # 지출 15% 절감 시 절약액
    debt_payoff_amount = round(debt * 0.2)           # 부채 20% 상환 금액
    income_increase_amount = round(income * 0.1)     # 소득 10% 증가액

    exp_red_prob = action_sims.get("expense_reduction", {}).get("probability", risk.get("probability"))
    debt_red_prob = action_sims.get("debt_reduction", {}).get("probability", risk.get("probability"))
    inc_inc_prob = action_sims.get("income_increase", {}).get("probability", risk.get("probability"))

    credit_lines = []
    if credit_score:
        credit_lines.append(f"- 신용점수: {credit_score}점 ({'양호' if credit_score >= 700 else '보통' if credit_score >= 600 else '주의 필요'})")
    if delinquency and delinquency != "없음":
        credit_lines.append(f"- 연체 이력: {delinquency} (신용 하락 위험 요소)")

    survival = risk.get("survival_years", 999)
    survival_str = "사실상 무한 (수입 > 지출)" if survival == 999 else f"{survival}년"

    prompt = f"""당신은 친근하고 전문적인 개인 재무 상담사입니다.
아래 고객의 재무 데이터를 분석하고, 마치 1:1 상담처럼 구체적이고 따뜻한 어조로 조언해 주세요.
반드시 JSON 형식으로만 응답하세요.

━━━ 고객 재무 현황 ━━━
• 연 수입: {income}만원 / 연 지출: {expense}만원
• 월 여유자금: {monthly_surplus}만원 {'(흑자)' if monthly_surplus >= 0 else '(적자 — 매달 부족)'}
• 총 부채: {debt}만원 / 총 자산: {assets}만원 / 순자산: {assets - debt}만원
• 부채 부담률: {risk.get('debt_ratio')} (연 수입의 {risk.get('debt_ratio')}배)
• 자금 유지 가능 기간: {survival_str}
{chr(10).join(credit_lines) if credit_lines else '• 신용 정보: 미입력'}

━━━ 리스크 진단 결과 ━━━
• 신용 리스크 확률: {risk.get('probability')}% (수준: {risk.get('level')})

━━━ 행동별 리스크 개선 효과 ━━━
• 연간 {expense_cut_amount}만원 지출 절감(15%) 시 → 리스크 {exp_red_prob}%
• 부채 {debt_payoff_amount}만원 상환(20%) 시 → 리스크 {debt_red_prob}%
• 연 수입 {income_increase_amount}만원 증가(10%) 시 → 리스크 {inc_inc_prob}%

━━━ 시나리오 분석 ━━━
{chr(10).join([f"• {s['name']}: 리스크 {s['result']['probability']}%" for s in scenarios])}

{chr(10).join(extra_sections)}

━━━ 응답 형식 (JSON만, 다른 텍스트 없이) ━━━
{{
  "summary": "고객에게 직접 말하는 듯한 어조로 현재 상황을 2~3문장으로 설명. 구체적 금액과 수치 포함. 신용정보·미래예측·은퇴분석 결과가 있으면 반드시 언급.",
  "risk_factors": [
    "리스크 요인 1 (구체적 수치 포함, 예: 부채가 연 수입의 3.7배로 상환 부담 과중)",
    "리스크 요인 2",
    "리스크 요인 3"
  ],
  "actions": [
    {{
      "title": "행동 제목 (간결하게)",
      "description": "마치 상담사가 직접 설명하듯 구체적으로. 어디서, 얼마를, 어떻게 줄이거나 늘릴지 명시. 월/연 금액 기준으로 설명.",
      "impact": "이 행동을 하면 리스크가 {risk.get('probability')}% → XX%로 줄어들어요. 연간 XX만원 절약 또는 부채 부담률 X.X → X.X로 개선 등 구체적 효과.",
      "priority": "high"
    }}
  ]
}}

작성 규칙:
1. summary와 description은 고객에게 직접 말하는 어투 (예: "현재 ~한 상황이에요", "~하시면 도움이 될 거예요")
2. actions는 3~5개, 가장 효과 큰 순서로 정렬
3. priority는 high/medium/low 중 하나
4. 모든 impact에 리스크 변화 수치 필수 포함
5. 신용점수가 낮거나 연체 이력이 있으면 신용 회복 action 반드시 포함"""

    try:
        content = _call_llm(prompt)
        return _parse_json(content)
    except Exception as e:
        print(f"[LLM ERROR] {type(e).__name__}: {e}")
        return _rule_based_advice(financial, risk, action_sims)


def _rule_based_advice(financial: dict, risk: dict, action_sims: dict) -> dict:
    """LLM 호출 실패 시 규칙 기반 조언 생성"""
    prob = risk.get("probability", 0)
    debt_ratio = risk.get("debt_ratio", 0)
    level = risk.get("level", "보통")

    actions = []

    exp_red = action_sims.get("expense_reduction", {}).get("probability", prob)
    if exp_red < prob:
        actions.append({
            "title": "지출 15% 절감",
            "description": "고정비 및 불필요한 소비를 줄여 월 지출을 15% 감축하세요.",
            "impact": f"리스크 {prob}% → {exp_red}%로 감소",
            "priority": "high",
        })

    debt_red = action_sims.get("debt_reduction", {}).get("probability", prob)
    if debt_red < prob:
        actions.append({
            "title": "부채 20% 상환",
            "description": "고금리 부채부터 우선 상환하여 부채 부담률을 낮추세요.",
            "impact": f"리스크 {prob}% → {debt_red}%로 감소",
            "priority": "high" if debt_ratio > 3 else "medium",
        })

    inc_inc = action_sims.get("income_increase", {}).get("probability", prob)
    if inc_inc < prob:
        actions.append({
            "title": "소득 10% 증대",
            "description": "부업, 역량 개발, 임금 협상 등을 통해 소득을 늘리세요.",
            "impact": f"리스크 {prob}% → {inc_inc}%로 감소",
            "priority": "medium",
        })

    risk_factors = []
    if debt_ratio > 3:
        risk_factors.append(f"부채 부담률 {debt_ratio} — 수입 대비 부채가 매우 높음")
    if financial.get("expense", 0) > financial.get("income", 1):
        risk_factors.append("지출이 수입을 초과 — 자산 잠식 중")
    delinquency = financial.get("delinquency", "없음")
    if delinquency and delinquency != "없음":
        risk_factors.append(f"연체 이력 {delinquency} — 신용 등급 하락 위험")
    credit_score = financial.get("credit_score")
    if credit_score and credit_score < 600:
        risk_factors.append(f"신용점수 {credit_score}점 — 대출 조건 불리")

    if not risk_factors:
        risk_factors.append("전반적 재무 지표 개선 필요")

    summary = f"현재 리스크 수준은 '{level}'({prob}%)입니다. "
    if debt_ratio > 3:
        summary += f"부채 부담률({debt_ratio})이 매우 높아 즉각적인 부채 관리가 필요합니다."
    else:
        summary += "지속적인 지출 관리와 소득 증대를 통해 리스크를 낮출 수 있습니다."

    return {
        "summary": summary,
        "risk_factors": risk_factors,
        "actions": actions if actions else [{"title": "재무 상담 권장", "description": "전문 재무 상담사와 상담을 권장합니다.", "impact": "리스크 개선 가능", "priority": "high"}],
    }


# ─────────────────────────────────────────
# 챗봇 응답 함수
# ─────────────────────────────────────────

def generate_chat_reply(message: str, context: dict) -> str:
    """분석 결과를 바탕으로 사용자 질문에 맞춤 답변 (일반 텍스트 반환)"""
    financial = context.get("input", context)
    risk = context.get("risk", {})
    advice = context.get("advice", {})

    income = financial.get("income", 0)
    expense = financial.get("expense", 0)
    debt = financial.get("debt", 0)
    assets = financial.get("assets", 0)
    credit_score = financial.get("credit_score")
    delinquency = financial.get("delinquency", "없음")

    try:
        monthly_surplus = round((income - expense) / 12, 1)
        net_worth = assets - debt
    except Exception:
        monthly_surplus = "미계산"
        net_worth = "미계산"

    survival_mode = risk.get("survival_mode", "")
    survival_years = risk.get("survival_years", "")
    if survival_mode == "debt_payoff":
        survival_str = f"부채 상환까지 {survival_years}년"
    elif survival_mode == "depletion":
        survival_str = f"자산 소진까지 {survival_years}년"
    else:
        survival_str = "안정적"

    extra_lines = []
    if credit_score:
        extra_lines.append(f"• 신용점수: {credit_score}점")
    if delinquency and delinquency != "없음":
        extra_lines.append(f"• 연체 이력: {delinquency}")
    for key, label in [
        ("income_growth_rate", "소득 증가율"),
        ("expense_growth_rate", "지출 증가율"),
        ("retirement_age", "은퇴 희망 나이"),
        ("dependents", "부양가족"),
        ("investment_return_rate", "투자 수익률"),
    ]:
        val = financial.get(key)
        if val:
            extra_lines.append(f"• {label}: {val}")

    prompt = f"""당신은 개인 재무 상담 전문가 AI입니다.
고객의 재무 데이터를 바탕으로 질문에 구체적이고 친근하게 답변하세요.

━━━ 고객 재무 현황 ━━━
• 연 수입: {income}만원 / 연 지출: {expense}만원
• 월 여유자금: {monthly_surplus}만원
• 총 부채: {debt}만원 / 총 자산: {assets}만원 / 순자산: {net_worth}만원
• 신용 리스크: {risk.get('probability', '미분석')}% ({risk.get('level', '')})
• 부채비율(자산 대비): {risk.get('debt_to_asset_pct', '미분석')}%
• 자금 상황: {survival_str}
{chr(10).join(extra_lines) if extra_lines else ''}

━━━ 기존 분석 요약 ━━━
{advice.get('summary', '분석 결과 없음')}

━━━ 고객 질문 ━━━
{message}

━━━ 답변 규칙 ━━━
1. 고객의 실제 수치(만원, %)를 직접 언급하며 맞춤 답변
2. 3~5문장으로 간결하게
3. 친근하고 전문적인 어투
4. 필요하면 구체적 실행 단계 포함
5. 반드시 일반 텍스트로만 답변 (JSON 금지)"""

    try:
        reply = _call_llm(prompt).strip()
        # JSON이 반환된 경우 텍스트 추출
        if reply.startswith("{"):
            import json as _json
            parsed = _json.loads(reply)
            reply = parsed.get("summary") or str(parsed)
        return reply
    except Exception:
        return "죄송합니다. 일시적인 오류가 발생했습니다. 잠시 후 다시 시도해 주세요."
