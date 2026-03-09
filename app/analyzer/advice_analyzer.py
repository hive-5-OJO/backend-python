import pandas as pd
from sqlalchemy import text

def calculate_advice_time_stats(ojo_engine):
    """
    [배치용] 시간대별 IN/OUT 상담 건수를 분리 집계합니다.
    """
    query = text("""
        SELECT 
            HOUR(created_at) AS hour, 
            direction, 
            COUNT(*) AS cnt
        FROM advice
        GROUP BY HOUR(created_at), direction
    """)
    
    df = pd.read_sql(query, con=ojo_engine)
    
    if df.empty: 
        return pd.DataFrame()

    stats = df.pivot_table(index='hour', columns='direction', values='cnt', fill_value=0)
    
    # 누락된 컬럼(IN/OUT) 보정 및 명칭 변경
    if 'IN' not in stats.columns: stats['IN'] = 0
    if 'OUT' not in stats.columns: stats['OUT'] = 0
    
    stats = stats.reset_index().rename(columns={'IN': 'inbound', 'OUT': 'outbound'})
    
    stats['total'] = stats['inbound'] + stats['outbound']
    
    return stats.sort_values(by='hour').reset_index(drop=True)

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