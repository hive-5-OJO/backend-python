import pandas as pd
import numpy as np
from sqlalchemy import text


def _pick_first_existing(df: pd.DataFrame, candidates: list[str]):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def calculate_churn_prediction(ojo_engine):
    query = text("""
        SELECT
            fc.*,
            fl.*,
            fm.*,
            fu.*
        FROM feature_consultation fc
        JOIN feature_lifecycle fl ON fc.member_id = fl.member_id
        JOIN feature_monetary fm ON fc.member_id = fm.member_id
        JOIN feature_usage fu ON fc.member_id = fu.member_id
    """)

    df = pd.read_sql(query, con=ojo_engine)

    if df.empty:
        empty_detail = pd.DataFrame(columns=["member_id", "churn_score", "risk_grade"])
        empty_summary = pd.DataFrame(columns=["grade", "count", "ratio"])
        return {
            "detail": empty_detail,
            "summary": empty_summary
        }

    # 컬럼 후보군
    consultation_col = _pick_first_existing(df, [
        "consultation_count", "consultation_cnt", "recent_consultation_count", "advice_count"
    ])
    inactive_days_col = _pick_first_existing(df, [
        "inactive_days", "dormant_days", "days_since_last_login", "days_since_last_usage"
    ])
    lifecycle_col = _pick_first_existing(df, [
        "subscription_months", "tenure_months", "membership_months", "lifecycle_score"
    ])
    monetary_col = _pick_first_existing(df, [
        "avg_monthly_amount", "monthly_amount", "total_payment_amount", "payment_amount"
    ])
    usage_col = _pick_first_existing(df, [
        "avg_usage", "usage_amount", "monthly_usage", "recent_usage"
    ])

    df["churn_score"] = 0

    # 1. 상담량 많으면 위험 증가
    if consultation_col:
        q75 = df[consultation_col].quantile(0.75)
        df.loc[df[consultation_col] >= q75, "churn_score"] += 25

    # 2. 비활성 일수 높으면 위험 증가
    if inactive_days_col:
        q75 = df[inactive_days_col].quantile(0.75)
        df.loc[df[inactive_days_col] >= q75, "churn_score"] += 35

    # 3. 사용량 낮으면 위험 증가
    if usage_col:
        q25 = df[usage_col].quantile(0.25)
        df.loc[df[usage_col] <= q25, "churn_score"] += 20

    # 4. 결제금액 낮으면 위험 증가
    if monetary_col:
        q25 = df[monetary_col].quantile(0.25)
        df.loc[df[monetary_col] <= q25, "churn_score"] += 20

    # 5. 가입 기간 짧으면 약간 위험
    if lifecycle_col:
        q25 = df[lifecycle_col].quantile(0.25)
        df.loc[df[lifecycle_col] <= q25, "churn_score"] += 10

    # 등급 분류
    df["risk_grade"] = "SAFE"
    df.loc[df["churn_score"] >= 40, "risk_grade"] = "WARNING"
    df.loc[df["churn_score"] >= 70, "risk_grade"] = "DANGER"

    # 상세 결과
    detail_cols = ["member_id", "churn_score", "risk_grade"]
    detail_df = df[detail_cols].copy()

    # 요약 결과
    total = len(detail_df)
    summary_rows = []
    for grade in ["DANGER", "WARNING", "SAFE"]:
        count = int((detail_df["risk_grade"] == grade).sum())
        ratio = round((count / total) * 100, 1) if total > 0 else 0.0
        summary_rows.append({
            "grade": grade,
            "count": count,
            "ratio": ratio
        })

    summary_df = pd.DataFrame(summary_rows)

    return {
        "detail": detail_df,
        "summary": summary_df
    }