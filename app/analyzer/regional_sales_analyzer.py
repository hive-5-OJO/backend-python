import pandas as pd
from sqlalchemy import text


def calculate_regional_sales(ojo_engine, analysis_engine):
    latest_date_query = "SELECT MAX(feature_base_date) FROM feature_monetary"

    with ojo_engine.connect() as conn:
        latest_date = conn.execute(text(latest_date_query)).scalar()

    if not latest_date:
        return []

    # 1. 원본 지역/매출/유형 데이터 조회 (ojo)
    region_query = """
    SELECT
        m.region,
        m.member_id,
        COALESCE(f.total_revenue, 0) AS total_revenue,
        COALESCE(f.monthly_revenue, 0) AS monthly_revenue,
        a.type
    FROM member m
    JOIN feature_monetary f
        ON m.member_id = f.member_id
    LEFT JOIN analysis a
        ON m.member_id = a.member_id
    WHERE f.feature_base_date = :latest_date
    """

    region_df = pd.read_sql(
        text(region_query),
        con=ojo_engine,
        params={"latest_date": latest_date}
    )

    if region_df.empty:
        return []

    # 2. 이탈 예측 결과 조회 (ojo_analysis)
    churn_query = """
    SELECT
        member_id,
        risk_grade
    FROM churn_prediction_snapshot
    """

    try:
        churn_df = pd.read_sql(text(churn_query), con=analysis_engine)
    except Exception:
        churn_df = pd.DataFrame(columns=["member_id", "risk_grade"])

    # 3. member_id 기준 병합
    merged_df = region_df.merge(
        churn_df,
        on="member_id",
        how="left"
    )

    # 예측 결과 없는 고객은 SAFE로 간주
    merged_df["risk_grade"] = merged_df["risk_grade"].fillna("SAFE")

    # 4. 지역별 집계
    stats = merged_df.groupby("region").apply(lambda x: pd.Series({
        "count": len(x),
        "ratio": round(len(x) / len(merged_df) * 100, 2) if len(merged_df) > 0 else 0,
        "totalRevenue": x["total_revenue"].sum(),
        "totalMonthlyRevenue": x["monthly_revenue"].sum(),
        "avgRevenue": x["total_revenue"].mean(),
        "avgMonthlyRevenue": x["monthly_revenue"].mean(),
        "vipCount": (x["type"] == "VIP").sum(),
        "churnRiskCount": (x["risk_grade"] == "DANGER").sum()
    })).reset_index()

    stats["churnRiskRatio"] = (
        stats["churnRiskCount"] / stats["count"].replace(0, 1) * 100
    ).round(2)

    return stats.to_dict(orient="records")