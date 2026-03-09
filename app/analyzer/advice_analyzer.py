import pandas as pd
from sqlalchemy import text

def calculate_advice_time_stats(ojo_engine):
    """
    [배치용] 전체 상담 내역을 가져와 시간대별로 카운트합니다.
    """
    # 자바 엔티티에 적힌 테이블명(advice)과 컬럼명(created_at)을 사용합니다.
    query = "SELECT created_at FROM advice"
    df = pd.read_sql(query, con=ojo_engine)
    
    if df.empty: return pd.DataFrame()

    df['created_at'] = pd.to_datetime(df['created_at'])
    df['hour'] = df['created_at'].dt.hour
    
    # 시간대별 카운트
    stats_df = df.groupby('hour').size().reset_index(name='adviceCount')
    stats_df = stats_df.sort_values(by='hour').reset_index(drop=True)
    
    return stats_df

def get_member_advice_timeline(ojo_engine, member_id):
    """
    [실시간용] 특정 고객의 상담 내역을 시간 역순으로 가져옵니다.
    """
    # 자바 엔티티의 컬럼명들을 그대로 가져옵니다.
    query = f"""
        SELECT 
            advice_id AS id, 
            direction, 
            channel, 
            advice_content AS adviceContent, 
            created_at AS createdAt, 
            satisfaction_score AS satisfactionScore
        FROM advice
        WHERE member_id = {member_id}
        ORDER BY created_at DESC
    """
    df = pd.read_sql(query, con=ojo_engine)
    
    # JSON 직렬화를 위해 날짜를 문자열로 변환
    if not df.empty:
        df['createdAt'] = df['createdAt'].astype(str)
        # NaN 값이 있으면 null로 처리
        df = df.where(pd.notnull(df), None)
        
    return df