import pandas as pd
import numpy as np

def calculate_ltv(ojo_engine):
    print("[분석] LTV(고객 생애 가치) 계산 중...")
    
    # 필요한 데이터 로드 
    query = """
    SELECT m.member_id, i.billed_amount, i.created_at 
    FROM member m
    JOIN invoice i ON m.member_id = i.member_id
    """
    df = pd.read_sql(query, con=ojo_engine)
    
    if df.empty: return pd.DataFrame()

    # LTV 계산 로직 (단순화: 평균 매출 * 구매 빈도 * 예상 유지 기간)
    ltv_base = df.groupby('member_id').agg({
        'billed_amount': ['mean', 'sum', 'count'],
        'created_at': [lambda x: (x.max() - x.min()).days]
    })
    ltv_base.columns = ['avg_value', 'total_revenue', 'frequency', 'lifespan_days']
    
    # 예측 LTV 계산 (예: 현재까지의 평균 매출에 유지 계수 적용)
    ltv_base['LTV'] = ltv_base['avg_value'] * ltv_base['frequency'] * 1.2 
    
    return ltv_base.reset_index()