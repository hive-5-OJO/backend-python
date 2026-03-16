import pandas as pd
import numpy as np
from sqlalchemy import text
from datetime import datetime

def calculate_rfm_metrics(ojo_engine):
    # DB에서 rfm 테이블 가져오기
    query = "SELECT member_id, recency, frequency, monetary FROM rfm"
    df = pd.read_sql(text(query), con=ojo_engine)

    if df.empty:
        return pd.DataFrame(columns=['member_id', 'rfm_score', 'type', 'lifecycle_stage'])

    # 기준일 설정 (파이프라인이 도는 현재 시간)
    now = datetime.now()
    df['recency'] = pd.to_datetime(df['recency'])

    # RFM 점수 계산 (1~5점 부여, 5점이 가장 좋음)
    df['R_rank'] = df['recency'].rank(method='first')
    df['F_rank'] = df['frequency'].rank(method='first')
    df['M_rank'] = df['monetary'].rank(method='first')

    df['R_score'] = pd.qcut(df['R_rank'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['F_score'] = pd.qcut(df['F_rank'], 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df['M_score'] = pd.qcut(df['M_rank'], 5, labels=[1, 2, 3, 4, 5]).astype(int)

    # 종합 rfm_score (100점 만점 환산)
    df['rfm_score'] = ((df['R_score'] * 0.2 + df['F_score'] * 0.4 + df['M_score'] * 0.4) / 5 * 100).astype(int)

    # type (고객 세그먼트) 계산
    type_conditions = [
        (df['R_score'] >= 4) & (df['F_score'] >= 4) & (df['M_score'] >= 4), # VIP
        (df['R_score'] >= 3) & (df['F_score'] >= 3) & (df['M_score'] >= 3), # LOYAL (잠재 VIP)
        (df['R_score'] <= 2) & (df['F_score'] >= 3),                        # RISK (이탈 위험 - 예전엔 단골이었음)
        (df['R_score'] <= 2) & (df['F_score'] <= 2)                         # LOST (이탈 - 오랫동안 안 옴)
    ]
    type_choices = ['VIP', 'LOYAL', 'RISK', 'LOST']
    df['type'] = np.select(type_conditions, type_choices, default='COMMON')

    # lifecycle_stage (생애주기) 계산
    df['days_since_last'] = (now - df['recency']).dt.days

    lc_conditions = [
        (df['days_since_last'] >= 90),                                 # CHURNED (이탈)
        (df['days_since_last'] >= 60),                                 # AT_RISK (이탈 위기)
        (df['days_since_last'] >= 30),                                 # STAGNANT (정체)
        (df['days_since_last'] < 30) & (df['F_score'] <= 2)            # NEW (신규 유입)
    ]
    lc_choices = ['CHURNED', 'AT_RISK', 'STAGNANT', 'NEW']
    df['lifecycle_stage'] = np.select(lc_conditions, lc_choices, default='GROWING') # GROWING (성장/활성 유지)

    # 최종 필요한 컬럼만 정리해서 반환
    result_df = df[['member_id', 'rfm_score', 'type', 'lifecycle_stage']]
    return result_df