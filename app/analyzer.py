import pandas as pd
import numpy as np

def analyze_all(dfs: dict, base_date: str):
    """
    순수 분석 로직만 담긴 파이썬 핵심 엔진입니다.
    데이터 출처와 무관하게 DataFrame 딕셔너리만 받아서 계산하고 결과를 반환합니다.
    """
    df_m, df_a, df_i = dfs['df_m'], dfs['df_a'], dfs['df_i']
    df_id, df_c, df_s, df_con = dfs['df_id'], dfs['df_c'], dfs['df_s'], dfs['df_con']
    
    today = pd.to_datetime(base_date) 
    
    # --- 1. RFM & LTV 분석 ---
    df_m['created_at_dt'] = pd.to_datetime(df_m['created_at'], errors='coerce')
    df_a['start_at_dt'] = pd.to_datetime(df_a['start_at'], errors='coerce')
    
    recency = df_a.groupby('member_id')['start_at_dt'].max().reset_index()
    recency['R'] = (today - recency['start_at_dt']).dt.days
    frequency = df_a.groupby('member_id').size().reset_index(name='F')
    monetary = df_i.groupby('member_id')['billed_amount'].sum().reset_index(name='M')
    
    rfm = df_m[['member_id', 'created_at', 'status', 'created_at_dt']].copy()
    rfm = rfm.merge(recency[['member_id', 'R']], on='member_id', how='left')
    rfm = rfm.merge(frequency, on='member_id', how='left')
    rfm = rfm.merge(monetary, on='member_id', how='left').fillna(0)
    
    rfm['R_pct'] = rfm['R'].rank(ascending=False, pct=True) * 100
    rfm['F_pct'] = rfm['F'].rank(ascending=True, pct=True) * 100
    rfm['M_pct'] = rfm['M'].rank(ascending=True, pct=True) * 100
    
    rfm['exact_score'] = (rfm['R_pct'] * 0.2) + (rfm['F_pct'] * 0.3) + (rfm['M_pct'] * 0.5)
    rfm['rfm_rank'] = rfm['exact_score'].rank(ascending=False, method='first').astype(int)
    rfm['top_percent'] = (rfm['rfm_rank'] / len(rfm) * 100).round(2)
    rfm.drop(columns=['R_pct', 'F_pct', 'M_pct'], inplace=True)
    
    raw_tenure = (today - rfm['created_at_dt']).dt.days / 30
    rfm['tenure_months'] = np.where(raw_tenure < 1, 1, raw_tenure)
    rfm['ARPU'] = rfm['M'] / rfm['tenure_months']
    
    total_canceled = len(df_s[df_s['status'] == 'CANCELED'])
    total_users = df_m['member_id'].nunique()
    avg_tenure = rfm['tenure_months'].mean()
    
    global_monthly_churn_rate = total_canceled / (total_users * avg_tenure) if total_users > 0 else 0.05
    if global_monthly_churn_rate < 0.01: 
        global_monthly_churn_rate = 0.01 
        
    lifespan_months = 1 / global_monthly_churn_rate
    rfm['LTV'] = (rfm['ARPU'] * lifespan_months).astype(int)
    rfm.drop(columns=['ARPU', 'tenure_months', 'created_at_dt'], inplace=True)
    rfm['created_at'] = rfm['created_at'].astype(str)
    
    # --- 2. 코호트 분석 ---
    df_m['SignupMonth'] = pd.to_datetime(df_m['created_at'], errors='coerce').dt.to_period('M')
    df_i['InvoiceMonth'] = pd.to_datetime(df_i['base_month'].astype(str).str.replace('-', ''), format='%Y%m', errors='coerce').dt.to_period('M')
    
    df_merged = pd.merge(df_i, df_m[['member_id', 'SignupMonth']], on='member_id')
    df_merged['CohortIndex'] = (df_merged['InvoiceMonth'] - df_merged['SignupMonth']).apply(lambda x: x.n if pd.notnull(x) else 0)
    cohort_data = df_merged.groupby(['SignupMonth', 'CohortIndex'])['member_id'].nunique().reset_index()
    cohort_pivot = cohort_data.pivot(index='SignupMonth', columns='CohortIndex', values='member_id')
    cohort_size = cohort_pivot.iloc[:, 0]
    retention = (cohort_pivot.divide(cohort_size, axis=0).round(3) * 100).fillna(0).reset_index()
    retention['SignupMonth'] = retention['SignupMonth'].astype(str)
    
    # --- 3. 대시보드 KPI ---
    top_plans = df_id[df_id['product_type'] == 'SUBSCRIPTION'].groupby('product_name_snapshot')['total_price'].sum().reset_index()
    top_plans = top_plans.sort_values(by='total_price', ascending=False).head(5)
    top_plans.columns = ['plan_name', 'total_revenue']
    
    df_cs = pd.merge(df_a, df_c, on='category_id', how='left')
    top_cs = df_cs['category_name'].value_counts().reset_index().head(5)
    top_cs.columns = ['category_name', 'inquiry_count']
    avg_sat = float(df_cs['satisfaction_score'].mean())
    
    df_subs_cancel = df_s[df_s['status'] == 'CANCELED']
    top_cancel_reasons = df_subs_cancel['reason_code'].value_counts().reset_index().head(3)
    top_cancel_reasons.columns = ['reason_code', 'cancel_count']
    
    subs_status = df_s['status'].value_counts(normalize=True).reset_index()
    subs_status.columns = ['status', 'ratio_percent']
    subs_status['ratio_percent'] = (subs_status['ratio_percent'] * 100).round(2)
    
    marketable = df_con[df_con['marketing_accepted'] == 'Y']
    active_marketable = pd.merge(marketable, df_m[df_m['status'] == 'ACTIVE'], on='member_id')
    marketable_count = len(active_marketable)
    marketable_ratio = (marketable_count / len(df_m)) * 100
    
    summary_data = [
        {"metric_name": "avg_cs_satisfaction", "metric_value": round(avg_sat, 2), "description": "전체 상담 평균 만족도"},
        {"metric_name": "avg_ltv", "metric_value": round(rfm['LTV'].mean(), 0), "description": "전체 고객 평균 LTV"},
        {"metric_name": "vip_ltv_cutoff", "metric_value": round(rfm['LTV'].quantile(0.90), 0), "description": "상위 10% VIP 커트라인"},
        {"metric_name": "marketable_target_count", "metric_value": marketable_count, "description": "프로모션 발송 가능 타겟 수"},
        {"metric_name": "marketable_target_ratio", "metric_value": round(marketable_ratio, 1), "description": "전체 대비 마케팅 동의 비율(%)"}
    ]
    df_summary = pd.DataFrame(summary_data)

    # --- 4. [신규] CS 상담 상세 통계 ---
    
    # 1) 시간대별 상담 통계 (start_at_dt에서 시간만 추출)
    df_a['hour'] = df_a['start_at_dt'].dt.hour
    time_stats = df_a['hour'].value_counts().reset_index()
    time_stats.columns = ['hour', 'inquiry_count']
    time_stats = time_stats.sort_values(by='hour')

    # 2) 만족도 분포 통계
    sat_stats = df_a['satisfaction_score'].value_counts(normalize=True).reset_index()
    sat_stats.columns = ['score', 'ratio_percent']
    sat_stats['ratio_percent'] = (sat_stats['ratio_percent'] * 100).round(2)
    sat_stats['count'] = df_a['satisfaction_score'].value_counts().values

    # 3) 상담사(Admin)별 성과 조회
    # (admin_id 컬럼이 df_a에 존재한다고 가정)
    if 'admin_id' in df_a.columns:
        admin_perf = df_a.groupby('admin_id').agg(
            total_calls=('advice_id', 'count'),
            avg_sat=('satisfaction_score', 'mean')
        ).reset_index()
        admin_perf['avg_sat'] = admin_perf['avg_sat'].round(2)
    else:
        admin_perf = pd.DataFrame() # 컬럼이 없으면 빈 테이블

    # 4) 지역별 매출 통계
    # (df_m에 address나 region 컬럼이 있다고 가정, rfm 테이블의 총 금액(M)과 결합)
    if 'region' in df_m.columns:
        region_df = pd.merge(df_m[['member_id', 'region']], rfm[['member_id', 'M']], on='member_id')
        region_stats = region_df.groupby('region').agg(
            user_count=('member_id', 'nunique'),
            total_revenue=('M', 'sum')
        ).reset_index().sort_values(by='total_revenue', ascending=False)
    else:
        region_stats = pd.DataFrame()

    # 5) 아웃바운드 성공률 통계 (통화 연결 상태 확정 반영!)
    # (direction 컬럼이 OUTBOUND인 것들 중, call_status의 비중을 구합니다)
    if 'call_status' in df_a.columns and 'direction' in df_a.columns:
        outbound_df = df_a[df_a['direction'].str.upper() == 'OUTBOUND']
        outbound_stats = outbound_df['call_status'].value_counts().reset_index()
        outbound_stats.columns = ['call_status', 'count']
    else:
        outbound_stats = pd.DataFrame()

    # 기존 리턴 값에 새로운 테이블들도 추가해서 내보냅니다!
    return {
        'rfm_snapshot': rfm,
        'cohort_snapshot': retention,
        'kpi_top_plans': top_plans,
        'kpi_top_cs': top_cs,
        'kpi_churn_reasons': top_cancel_reasons,
        'kpi_subs_status': subs_status,
        'kpi_summary_metrics': df_summary,
        # 새로 추가된 테이블들
        'kpi_advice_time': time_stats,
        'kpi_advice_sat': sat_stats,
        'kpi_admin_perf': admin_perf,
        'kpi_region_stats': region_stats,
        'kpi_advice_outbound': outbound_stats
    }