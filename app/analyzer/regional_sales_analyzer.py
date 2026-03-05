import pandas as pd
from sqlalchemy import text

def calculate_regional_sales(ojo_engine):
    latest_date_query = "SELECT MAX(feature_base_date) FROM feature_monetary"
    with ojo_engine.connect() as conn:
        latest_date = conn.execute(text(latest_date_query)).scalar()

    if not latest_date: return []
    
    query = """
    SELECT
        m.region, 
        m.member_id, 
        COALESCE(f.total_revenue, 0) as total_revenue,
        COALESCE(f.monthly_revenue, 0) as monthly_revenue,
        a.type,
        cp.risk_grade
    FROM member m JOIN feature_monetary f ON m.member_id = f.member_id
        LEFT JOIN analysis a ON m.member_id = a.member_id
        LEFT JOIN churn_prediction cp ON m.member_id = cp.member_id
    WHERE f.feature_base_date = :latest_date
    """
    df = pd.read_sql(text(query), con=ojo_engine, params={"latest_date": latest_date})
    
    if df.empty: return pd.DataFrame()

    # 지역별 고객 분포 및 매출 - 성능 개선을 위해 agg 사용할 수도 있음
    stats =  df.groupby('region').apply(lambda x : pd.Series({
        "count": len(x),
        "ratio": round(len(x) / len(df) * 100, 2) if len(df) > 0 else 0,
        "totalRevenue": x['total_revenue'].sum(),
        "totalMonthlyRevenue": x['monthly_revenue'].sum(),
        "avgRevenue" : x['total_revenue'].mean(),
        "avgMonthlyRevenue" : x['monthly_revenue'].mean(),

        "vipCount": (x['type'] == 'VIP').sum(),
        "churnRiskCount": (x['risk_grade'] == 'DANGER').sum()
    })).reset_index()

    stats['churnRiskRatio'] = (
        stats['churnRiskCount'] / stats['count'].replace(0, 1) * 100 
    )
    
    return stats.to_dict(orient='records')