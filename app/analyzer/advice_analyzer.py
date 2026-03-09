import pandas as pd
from sqlalchemy import text

def calculate_advice_time_stats(ojo_engine):
    """
    [배치용] 시간대별 IN/OUT 상담 건수를 분리 집계합니다.
    """
    # direction 컬럼을 함께 가져와야 인바운드/아웃바운드 구분이 가능합니다.
    query = "SELECT created_at, direction FROM advice"
    df = pd.read_sql(query, con=ojo_engine)
    
    if df.empty: return pd.DataFrame()

    df['created_at'] = pd.to_datetime(df['created_at'])
    df['hour'] = df['created_at'].dt.hour
    
    # 1. 시간대와 방향으로 그룹화하여 카운트
    stats = df.groupby(['hour', 'direction']).size().unstack(fill_value=0)
    
    # 2. 누락된 컬럼(IN/OUT) 보정 및 명칭 변경
    if 'IN' not in stats.columns: stats['IN'] = 0
    if 'OUT' not in stats.columns: stats['OUT'] = 0
    
    stats = stats.reset_index().rename(columns={'IN': 'inbound', 'OUT': 'outbound'})
    
    # 3. 전체 합계(total) 계산
    stats['total'] = stats['inbound'] + stats['outbound']
    
    return stats.sort_values(by='hour').reset_index(drop=True)

def get_member_advice_timeline(ojo_engine, member_id):
    """
    [실시간용] 특정 고객의 타임라인을 명세서 규격에 맞춰 가져옵니다.
    """
    # promotionName 조인을 위해 LEFT JOIN을 사용합니다.
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
        WHERE a.member_id = {member_id}
        ORDER BY a.created_at DESC
    """
    df = pd.read_sql(query, con=ojo_engine)
    
    if not df.empty:
        # 날짜 형식을 YYYY-MM-DD로 변환
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
        df = df.where(pd.notnull(df), None)
        
    return df