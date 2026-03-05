import pandas as pd

def calculate_cohort(ojo_engine):
    print("[분석] 코호트(유지율) 분석 중...")
    
    # 1. 회원 가입일과 결제 이력 가져오기
    query = """
    SELECT m.member_id, m.created_at as join_date, i.created_at as order_date
    FROM member m
    JOIN invoice i ON m.member_id = i.member_id
    """
    df = pd.read_sql(query, con=ojo_engine)
    if df.empty: return pd.DataFrame()

    df['join_month'] = df['join_date'].dt.to_period('M')
    df['order_month'] = df['order_date'].dt.to_period('M')
    df['cohort_index'] = (df['order_month'] - df['join_month']).apply(lambda x: x.n)

    cohort_data = df.groupby(['join_month', 'cohort_index'])['member_id'].nunique().reset_index()
    cohort_pivot = cohort_data.pivot(index='join_month', columns='cohort_index', values='member_id')
    
    # 첫 달 기준으로 비율 계산
    cohort_size = cohort_pivot.iloc[:, 0]
    retention = cohort_pivot.divide(cohort_size, axis=0)
    
    # DB 저장을 위해 가입월을 문자열로 변경
    retention.index = retention.index.astype(str)
    return retention.reset_index()