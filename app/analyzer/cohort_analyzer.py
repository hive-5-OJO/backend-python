import pandas as pd
import numpy as np

def calculate_segmented_cohort(ojo_engine, segment_type='all'):
    print(f"[분석] {segment_type} 기준 코호트 분석 중...")
    
    # 1. 원천 데이터 로드 (쿼리 부분 인덴트 정렬)
    query = """
    SELECT 
        m.member_id, 
        m.created_at as join_date, 
        i.created_at as order_date,
        c.total_consult_count as consult_count,
        mon.total_revenue as total_amount, 
        r.monetary as rfm_monetary,  -- grade 대신 rfm 테이블의 monetary를 가져옴
        r.frequency as rfm_frequency
    FROM member m
    JOIN invoice i ON m.member_id = i.member_id
    LEFT JOIN feature_consultation c ON m.member_id = c.member_id
    LEFT JOIN feature_monetary mon ON m.member_id = mon.member_id
    LEFT JOIN rfm r ON m.member_id = r.member_id
    """
    
    df = pd.read_sql(query, con=ojo_engine)
    
    if df.empty: 
        print("[경고] 조회된 데이터가 없습니다.")
        return pd.DataFrame()

    # 데이터 타입 변환 및 결측치 처리
    df['join_date'] = pd.to_datetime(df['join_date'])
    df['order_date'] = pd.to_datetime(df['order_date'])
    df['consult_count'] = df['consult_count'].fillna(0)
    df['total_amount'] = df['total_amount'].fillna(0)

    # 2. 세그먼트 필터링 로직 
    if segment_type == 'high_consult':
        df = df[df['consult_count'] >= 5]
    elif segment_type == 'vip':
        df = df[df['rfm_monetary'] >= 1000000] 
    elif segment_type == 'big_spender':
        df = df[df['total_amount'] >= 1000000]

    if df.empty:
        print(f"[정보] {segment_type} 조건에 맞는 데이터가 없습니다.")
        return pd.DataFrame()

    # 3. 코호트 인덱스 계산
    df['join_month'] = df['join_date'].dt.to_period('M')
    df['order_month'] = df['order_date'].dt.to_period('M')
    df['cohort_index'] = (df['order_month'] - df['join_month']).apply(lambda x: x.n)

    # 4. 피벗 테이블 및 유지율 계산
    cohort_data = df.groupby(['join_month', 'cohort_index'])['member_id'].nunique().reset_index()
    cohort_pivot = cohort_data.pivot(index='join_month', columns='cohort_index', values='member_id')
    
    # 첫 달 기준으로 비율 계산
    cohort_size = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_size, axis=0)
    
    # DB 저장을 위해 가입월을 문자열로 변경
    retention.index = retention.index.astype(str)
    result_df = retention.reset_index()
    
    result_df['segment_type'] = segment_type 
    return result_df