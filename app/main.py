from fastapi import FastAPI, BackgroundTasks
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import io
import re

# 우리가 만든 database.py 파일에서 engine 불러오기
from database import engine

# 윈도우 터미널 한글 깨짐 방지 설정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

app = FastAPI(title="High-5 Data Analysis Batch Server (1-Year Full Data)")

# ==========================================
# [함수 1] 다중 파일 자동 검색 및 병합(Concat) 로직
# ==========================================
def find_single_csv(base_path, filename):
    for root, dirs, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

def find_and_concat_monthly_csvs(base_path, pattern):
    df_list = []
    found_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if re.match(pattern, file):
                file_path = os.path.join(root, file)
                try:
                    df_list.append(pd.read_csv(file_path))
                    found_files.append(file)
                except Exception as e:
                    print(f"[경고] {file} 읽기 실패: {e}")
                    
    if df_list:
        print(f" -> {len(found_files)}개의 파일을 찾아 병합했습니다. (예: {found_files[0]} 등)")
        return pd.concat(df_list, ignore_index=True)
    return None

def load_real_data():
    print("\n[데이터 로딩] 전체 폴더를 스캔하여 1년 치(12개월) CSV 파일을 자동 병합합니다...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    dfs = {}
    print("- 청구서 상세(invoice_detail) 로딩 중...")
    dfs['df_id'] = find_and_concat_monthly_csvs(base_dir, r'^invoice_detail_2026\d{2}\.csv$')
    
    print("- 청구서(invoice) 로딩 중...")
    dfs['df_i'] = find_and_concat_monthly_csvs(base_dir, r'^invoice_2026\d{2}\.csv$')
    
    print("- 상담 내역(advice) 로딩 중...")
    dfs['df_a'] = find_and_concat_monthly_csvs(base_dir, r'^advice_2026\d{2}\.csv$')
    
    print("- [New] 결제 내역(payment) 로딩 중...")
    dfs['df_p'] = find_and_concat_monthly_csvs(base_dir, r'^payment_2026\d{2}\.csv$')
    
    print("- [New] 데이터 사용량(data_usage) 로딩 중...")
    dfs['df_du'] = find_and_concat_monthly_csvs(base_dir, r'^data_usage_2026\d{2}\.csv$')

    single_files = {
        'df_c': 'categories.csv',
        'df_s': 'subscription_period.csv',
        'df_con': 'member_consent.csv',
        'df_m': 'member.csv'
    }
    
    for key, filename in single_files.items():
        found_path = find_single_csv(base_dir, filename)
        if found_path:
            dfs[key] = pd.read_csv(found_path)
        else:
            print(f"[경고] 단일 파일 누락: {filename}")
            return None

    # ★ 수정됨: 파일이 없어도 배치가 멈추지 않도록 예외 처리
    for key, df in list(dfs.items()):
        if df is None or df.empty:
            if key in ['df_p', 'df_du']:
                print(f"👉 [안내] {key} 파일이 없거나 비어있지만, 기존 분석을 위해 무시하고 넘어갑니다.")
                dfs[key] = pd.DataFrame() # 에러 방지용 빈 껍데기
            else:
                print(f"[오류] 필수 데이터({key})를 불러오지 못해 분석을 중단합니다.")
                return None

    print("\n✅ 모든 데이터 로딩 단계를 통과했습니다!\n")
    return dfs

# ==========================================
# [함수 2] 배치 분석 엔진 (RFM 순위 + 대시보드 KPI 요약 -> DB 저장)
# ==========================================
def perform_full_analysis_and_save():
    dfs = load_real_data()
    if not dfs:
        return
        
    print("[배치 실행] 거대한 1년 치 데이터 분석 및 DB 저장을 시작합니다...")
    df_m, df_a, df_i = dfs['df_m'], dfs['df_a'], dfs['df_i']
    df_id, df_c, df_s, df_con = dfs['df_id'], dfs['df_c'], dfs['df_s'], dfs['df_con']
    
    # 2026년 12월까지의 데이터이므로, 기준일을 2027년 1월 1일로 설정
    today = pd.to_datetime('2027-01-01') 
    
    # ----------------------------------------------------
    # [파트 A] 개별 고객 상세 분석 (RFM 세부 랭킹 & LTV, 코호트)
    # ----------------------------------------------------
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
    
    rfm['tenure_months'] = (today - rfm['created_at_dt']).dt.days // 30
    rfm['LTV'] = (rfm['M'] * rfm['tenure_months']).astype(int)
    
    # ★ 수정됨: 코호트 기준을 created_at(오류 원인)이 아닌 base_month(청구 기준월)로 변경
    df_m['SignupMonth'] = df_m['created_at_dt'].dt.to_period('M')
    df_i['InvoiceMonth'] = pd.to_datetime(df_i['base_month'].astype(str).str.replace('-', ''), format='%Y%m', errors='coerce').dt.to_period('M')
    
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
    
    print("분석 완료! 거대한 데이터를 DB에 밀어 넣습니다...")
    
    rfm.to_sql(name='rfm_snapshot', con=engine, if_exists='replace', index=False)
    retention.to_sql(name='cohort_snapshot', con=engine, if_exists='replace', index=False)
    top_plans.to_sql(name='kpi_top_plans', con=engine, if_exists='replace', index=False)
    top_cs.to_sql(name='kpi_top_cs', con=engine, if_exists='replace', index=False)
    top_cancel_reasons.to_sql(name='kpi_churn_reasons', con=engine, if_exists='replace', index=False)
    subs_status.to_sql(name='kpi_subs_status', con=engine, if_exists='replace', index=False)
    df_summary.to_sql(name='kpi_summary_metrics', con=engine, if_exists='replace', index=False)
    
    print("[배치 완료] 1년 치 전체 데이터 분석 및 DB 저장이 완벽하게 종료되었습니다.")

# ==========================================
# [API 엔드포인트]
# ==========================================

@app.post("/api/analysis/trigger-batch")
async def trigger_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(perform_full_analysis_and_save)
    return {"status": "started", "message": "1년 치 전체 데이터 분석이 시작되었습니다."}

@app.get("/api/analysis/db-check")
def check_rfm_ltv_data():
    try:
        df = pd.read_sql("SELECT member_id, exact_score, rfm_rank, top_percent, LTV FROM rfm_snapshot ORDER BY rfm_rank ASC LIMIT 5", con=engine)
        return {"top_5_vip_sample": df.fillna(0).to_dict(orient='records')}
    except Exception as e:
        return {"message": "데이터가 없습니다. 먼저 배치를 실행하세요.", "error": str(e)}

@app.get("/api/analysis/cohort-check")
def check_cohort_data():
    try:
        # 새로 추가: 코호트 데이터를 직접 브라우저에서 볼 수 있게 추가했습니다.
        df = pd.read_sql("SELECT * FROM cohort_snapshot", con=engine)
        return {"cohort_data": df.fillna(0).to_dict(orient='records')}
    except Exception as e:
        return {"message": "데이터가 없습니다.", "error": str(e)}

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