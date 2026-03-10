import pandas as pd
from sqlalchemy import text

def get_member_advice_timeline(ojo_engine, member_id):
    """
    [실시간용] 특정 고객의 타임라인을 명세서 규격에 맞춰 가져옵니다.
    """
    query = f"""
        SELECT 
            a.advice_id AS id, 
            a.created_at AS date, 
            c.category_name AS category,
            a.direction, 
            a.advice_content AS content, 
            p.promotion_name AS promotionName,
            a.satisfaction_score AS satisfactionScore
        FROM advice a
        LEFT JOIN categories c ON a.category_id = c.category_id
        LEFT JOIN promotion p ON a.promotion_id = p.promotion_id
        WHERE a.member_id = :member_id
        ORDER BY a.created_at DESC
    """
    df = pd.read_sql(query, con=ojo_engine, params={"member_id": member_id})
    
    if not df.empty:
        # 날짜 형식을 YYYY-MM-DD로 변환
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df = df.where(pd.notnull(df), None)
        
    return df