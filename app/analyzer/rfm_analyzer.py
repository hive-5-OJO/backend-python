import pandas as pd
import numpy as np
from sqlalchemy import text
from datetime import datetime

def calculate_rfm_metrics(ojo_engine):
    query = "SELECT member_id, recency, frequency, monetary FROM rfm"
    df = pd.read_sql(text(query), con=ojo_engine)

    if df.empty:
        return pd.DataFrame(columns=['member_id', 'rfm_score', 'type', 'lifecycle_stage'])

    now = datetime.now()
    df['recency'] = pd.to_datetime(df['recency'])

    # RFM 점수 계산 (1~5점 부여, 5점이 가장 좋음)
    df['R_rank'] = df['recency'].rank(method='first')
    df['F_rank'] = df['frequency'].rank(method='first')
    df['M_rank'] = df['monetary'].rank(method='first')

    # qcut으로 상위 20%씩 잘라서 1~5점 부여
    df['R_score'] = pd.qcut(df['R_rank'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['F_score'] = pd.qcut(df['F_rank'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['M_score'] = pd.qcut(df['M_rank'], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    # 종합 rfm_score (100점 만점 환산: R 20%, F 40%, M 40% 비중)
    df['rfm_score'] = ((df['R_score'] * 0.2 + df['F_score'] * 0.4 + df['M_score'] * 0.4) / 5 * 100).astype(int)

    # type (고객 세그먼트) 계산
    conditions = [
        (df['F_score'] >= 4) & (df['M_score'] >= 4) & (df['R_score'] >= 4), # 최근에도 오고 돈도 많이 씀
        (df['F_score'] >= 4) & (df['M_score'] >= 4),                        # 누적 결제/방문이 많음
        (df['R_score'] >= 4) & (df['F_score'] <= 2),                        # 최근에 처음 가입/결제함
        (df['R_score'] <= 2) & (df['F_score'] >= 3)                         # 예전엔 많이 샀는데 요새 안 옴
    ]
    choices = ['VIP', '우수', '신규', '이탈위험']
    df['type'] = np.select(conditions, choices, default='일반')

    # lifecycle_stage (생애주기) 계산
    df['days_since_last'] = (now - df['recency']).dt.days

    lc_conditions = [
        (df['days_since_last'] <= 30), # 1달 이내 접속/결제
        (df['days_since_last'] <= 90)  # 3달 이내 접속/결제
    ]
    lc_choices = ['ACTIVE', 'AT_RISK']
    df['lifecycle_stage'] = np.select(lc_conditions, lc_choices, default='DORMANT') # 3달 넘어가면 휴면

    result_df = df[['member_id', 'rfm_score', 'type', 'lifecycle_stage']]
    return result_df