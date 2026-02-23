from fastapi import FastAPI, BackgroundTasks
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import io

# 우리가 만든 database.py 파일에서 engine 불러오기
from database import engine

# 윈도우 터미널 한글 깨짐 방지 설정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

app = FastAPI(title="High-5 Data Analysis Batch Server (Integrated & Ranked)")

# ==========================================
# [함수 1] 파일 자동 검색 및 로드 로직
# ==========================================
def find_csv_file(base_path, filename):
    for root, dirs, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

def load_real_data():
    print("[데이터 로딩] 전체 폴더를 스캔하여 CSV 파일을 자동 검색합니다...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    required_files = {
        'invoice_detail': 'invoice_detail_202601.csv',
        'invoice': 'invoice_202601.csv',
        'advice': 'advice_202601.csv',
        'categories': 'categories.csv',
        'subs': 'subscription_period.csv',
        'consent': 'member_consent.csv',
        'member': 'member.csv'
    }
    
    file_paths = {}
    missing_files = []
    
    for key, filename in required_files.items():
        found_path = find_csv_file(base_dir, filename)
        if found_path:
            file_paths[key] = found_path
        else:
            missing_files.append(filename)
            
    if missing_files:
        print("경고: 다음 파일을 찾을 수 없습니다:", missing_files)
        return None

    # 데이터 로드
    dfs = {
        'df_id': pd.read_csv(file_paths['invoice_detail']),
        'df_i': pd.read_csv(file_paths['invoice']),
        'df_a': pd.read_csv(file_paths['advice']),
        'df_c': pd.read_csv(file_paths['categories']),
        'df_s': pd.read_csv(file_paths['subs']),
        'df_con': pd.read_csv(file_paths['consent']),
        'df_m': pd.read_csv(file_paths['member'])
    }
    print("모든 CSV 파일을 성공적으로 로드했습니다.")
    return dfs

# ==========================================
# [함수 2] 배치 분석 엔진 (RFM 순위 + 대시보드 KPI 요약 -> DB 저장)
# ==========================================
def perform_full_analysis_and_save():
    print("\n[배치 실행] 데이터 분석 및 DB 저장을 시작합니다...")
    dfs = load_real_data()
    if not dfs:
        print("[오류] 파일 누락으로 분석을 중단합니다.")
        return
        
    df_m = dfs['df_m']
    df_a = dfs['df_a']
    df_i = dfs['df_i']
    df_id = dfs['df_id']
    df_c = dfs['df_c']
    df_s = dfs['df_s']
    df_con = dfs['df_con']
    
    today = pd.to_datetime('2026-02-14') # 분석 기준일 설정
    
    # ----------------------------------------------------
    # [파트 A] 개별 고객 상세 분석 (RFM 세부 랭킹 & LTV, 코호트)
    # ----------------------------------------------------
    # 날짜 데이터 변환
    df_m['created_at_dt'] = pd.to_datetime(df_m['created_at'], errors='coerce')
    df_a['start_at_dt'] = pd.to_datetime(df_a['start_at'], errors='coerce')
    
    # 1. R, F, M 기초 데이터 계산
    recency = df_a.groupby('member_id')['start_at_dt'].max().reset_index()
    recency['R'] = (today - recency['start_at_dt']).dt.days
    frequency = df_a.groupby('member_id').size().reset_index(name='F')
    monetary = df_i.groupby('member_id')['billed_amount'].sum().reset_index(name='M')
    
    rfm = df_m[['member_id', 'created_at', 'status', 'created_at_dt']].copy()
    rfm = rfm.merge(recency[['member_id', 'R']], on='member_id', how='left')
    rfm = rfm.merge(frequency, on='member_id', how='left')
    rfm = rfm.merge(monetary, on='member_id', how='left').fillna(0)
    
    # ★ 수정된 부분: 백분위수 기반의 정밀한 스코어링 및 등수 매기기
    # R(최근성)은 숫자가 작을수록 좋음 (ascending=False)
    rfm['R_pct'] = rfm['R'].rank(ascending=False, pct=True) * 100
    # F(빈도), M(금액)은 숫자가 클수록 좋음 (ascending=True)
    rfm['F_pct'] = rfm['F'].rank(ascending=True, pct=True) * 100
    rfm['M_pct'] = rfm['M'].rank(ascending=True, pct=True) * 100
    
    # 가중치 부여 (R 20%, F 30%, M 50% 적용)
    rfm['exact_score'] = (rfm['R_pct'] * 0.2) + (rfm['F_pct'] * 0.3) + (rfm['M_pct'] * 0.5)
    
    # 등수(Rank) 및 상위 퍼센트(%) 계산 (method='first'로 동점자 방지)
    rfm['rfm_rank'] = rfm['exact_score'].rank(ascending=False, method='first').astype(int)
    rfm['top_percent'] = (rfm['rfm_rank'] / len(rfm) * 100).round(2)
    
    # 필요 없는 임시 컬럼(백분위 점수)은 깔끔하게 지워줍니다.
    rfm.drop(columns=['R_pct', 'F_pct', 'M_pct'], inplace=True)
    
    # LTV 계산
    rfm['tenure_months'] = (today - rfm['created_at_dt']).dt.days // 30
    rfm['LTV'] = (rfm['M'] * rfm['tenure_months']).astype(int)
    
    # 2. 코호트 분석 (유지율)
    df_m['SignupMonth'] = df_m['created_at_dt'].dt.to_period('M')
    df_i['InvoiceMonth'] = pd.to_datetime(df_i['created_at'], errors='coerce').dt.to_period('M')
    df_merged = pd.merge(df_i, df_m[['member_id', 'SignupMonth']], on='member_id')
    df_merged['CohortIndex'] = (df_merged['InvoiceMonth'] - df_merged['SignupMonth']).apply(lambda x: x.n if pd.notnull(x) else 0)
    cohort_data = df_merged.groupby(['SignupMonth', 'CohortIndex'])['member_id'].nunique().reset_index()
    cohort_pivot = cohort_data.pivot(index='SignupMonth', columns='CohortIndex', values='member_id')
    cohort_size = cohort_pivot.iloc[:, 0]
    retention = (cohort_pivot.divide(cohort_size, axis=0).round(3) * 100).fillna(0).reset_index()
    
    # ----------------------------------------------------
    # [파트 B] 대시보드용 KPI 통계 분석
    # ----------------------------------------------------
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

    # ----------------------------------------------------
    # [파트 C] DB에 모든 데이터 저장하기
    # ----------------------------------------------------
    rfm.drop(columns=['created_at_dt'], inplace=True, errors='ignore')
    rfm['created_at'] = rfm['created_at'].astype(str)
    retention['SignupMonth'] = retention['SignupMonth'].astype(str)
    
    print("분석 완료! DB에 테이블들을 생성하고 저장합니다...")
    
    rfm.to_sql(name='rfm_snapshot', con=engine, if_exists='replace', index=False)
    retention.to_sql(name='cohort_snapshot', con=engine, if_exists='replace', index=False)
    top_plans.to_sql(name='kpi_top_plans', con=engine, if_exists='replace', index=False)
    top_cs.to_sql(name='kpi_top_cs', con=engine, if_exists='replace', index=False)
    top_cancel_reasons.to_sql(name='kpi_churn_reasons', con=engine, if_exists='replace', index=False)
    subs_status.to_sql(name='kpi_subs_status', con=engine, if_exists='replace', index=False)
    df_summary.to_sql(name='kpi_summary_metrics', con=engine, if_exists='replace', index=False)
    
    print("[배치 완료] 상세 데이터 및 KPI 통계가 모두 DB에 성공적으로 저장되었습니다.")

# ==========================================
# [API 엔드포인트]
# ==========================================

@app.post("/api/analysis/trigger-batch")
async def trigger_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(perform_full_analysis_and_save)
    return {"status": "started", "message": "랭킹 시스템이 반영된 통합 배치 분석이 백그라운드에서 시작되었습니다."}

@app.get("/api/analysis/db-check")
def check_rfm_ltv_data():
    try:
        # 수정됨: 조회할 때 기존 rfm_score 대신 exact_score, rfm_rank, top_percent를 가져옵니다.
        df = pd.read_sql("SELECT member_id, exact_score, rfm_rank, top_percent, LTV FROM rfm_snapshot ORDER BY rfm_rank ASC LIMIT 5", con=engine)
        return {"top_5_vip_sample": df.fillna(0).to_dict(orient='records')}
    except Exception as e:
        return {"message": "데이터가 없습니다. 먼저 배치를 실행하세요.", "error": str(e)}

@app.get("/api/analysis/kpi-check")
def check_kpi_summary():
    try:
        df = pd.read_sql("SELECT * FROM kpi_summary_metrics", con=engine)
        return {"kpi_summary_metrics": df.fillna(0).to_dict(orient='records')}
    except Exception as e:
        return {"message": "KPI 데이터가 없습니다. 먼저 배치를 실행하세요.", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)