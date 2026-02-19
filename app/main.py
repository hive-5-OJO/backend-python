from fastapi import FastAPI, BackgroundTasks
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy import create_engine

app = FastAPI(title="High-5 Analysis Batch Server (Final Fixed)")

# 1. 로컬 DB 연결 (sqlite 파일)
DB_URL = "sqlite:///./analysis_data.db"
engine = create_engine(DB_URL)


# 더미 데이터 생성 (수정판: 가입일 기준 결제 생성)
def generate_instant_dummy():
    np.random.seed(42)
    n = 100
    today = datetime(2026, 2, 14)
    
    # 1. 회원 데이터 생성
    segments = ['장기 VIP', '데이터 헤비', '가격 민감', '잦은 민원', '일반']
    member_data = []
    for i in range(n):
        seg = np.random.choice(segments)
        # 가입일을 다양하게 (최근 1년치 집중)
        join_date = today - timedelta(days=np.random.randint(1, 365))
        member_data.append({
            'customer_id': 1000 + i,
            'segment': seg,
            'created_at': join_date
        })
    df_m = pd.DataFrame(member_data)
    
    # 2. 상담 데이터 생성
    advice_data = []
    for i in range(n):
        count = np.random.randint(1, 10) # 누구나 1번쯤은 상담함
        for _ in range(count):
            advice_data.append({
                'customer_id': 1000 + i,
                'start_at': df_m.iloc[i]['created_at'] + timedelta(days=np.random.randint(0, 100))
            })
    df_a = pd.DataFrame(advice_data)
    
    # 3. 매출(Invoice) 데이터 생성
    invoice_data = []
    for i in range(n):
        signup_date = df_m.iloc[i]['created_at']
        seg = df_m.iloc[i]['segment']
        base_pay = 80000 if seg == '데이터 헤비' else 40000
        
        # 가입 직후부터 매달 결제한다고 가정 (최대 12개월)
        # 랜덤하게 중도 이탈(range를 줄임)하는 효과 추가
        retention_months = np.random.randint(1, 13) 
        
        for m in range(retention_months):
            pay_date = signup_date + timedelta(days=m*30)
            if pay_date > today: break # 미래 날짜 방지
            
            invoice_data.append({
                'customer_id': 1000 + i,
                'billed_amount': base_pay + np.random.randint(-5000, 5000),
                'paid_at': pay_date
            })
            
    df_i = pd.DataFrame(invoice_data)
            
    return df_m, df_a, df_i

# 배치 분석 엔진 (RFM + LTV + Cohort -> DB 저장)
def perform_full_analysis_and_save():
    print("🚀 [배치] 데이터 분석 시작...")
    df_m, df_a, df_i = generate_instant_dummy()
    today = datetime(2026, 2, 14)
    
    # RFM & LTV 계산
    recency = df_a.groupby('customer_id')['start_at'].max().reset_index()
    recency['R'] = (today - recency['start_at']).dt.days
    frequency = df_a.groupby('customer_id').size().reset_index(name='F')
    monetary = df_i.groupby('customer_id')['billed_amount'].sum().reset_index(name='M')
    
    # 데이터 병합 (NaN 방지를 위해 fillna(0) 즉시 적용)
    rfm = df_m.merge(recency, on='customer_id', how='left') \
              .merge(frequency, on='customer_id', how='left') \
              .merge(monetary, on='customer_id', how='left') \
              .fillna(0)
    
    # RFM 스코어링
    rfm['R_score'] = pd.qcut(rfm['R'].rank(method='first'), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    rfm['F_score'] = pd.qcut(rfm['F'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['M_score'] = pd.qcut(rfm['M'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm['total_score'] = rfm['R_score'] + rfm['F_score'] + rfm['M_score']
    
    #LTV (고객 생애 가치) 계산
    rfm['tenure_months'] = (today - rfm['created_at']).dt.days // 30
    rfm['avg_monthly_revenue'] = rfm['M'] / 3
    rfm['LTV'] = (rfm['avg_monthly_revenue'] * rfm['tenure_months']).astype(int)
    
    #  코호트(Cohort) 분석
    df_m['SignupMonth'] = pd.to_datetime(df_m['created_at']).dt.to_period('M')
    df_i['InvoiceMonth'] = pd.to_datetime(df_i['paid_at']).dt.to_period('M')
    
    df_merged = pd.merge(df_i, df_m[['customer_id', 'SignupMonth']], on='customer_id')
    df_merged['CohortIndex'] = (df_merged['InvoiceMonth'] - df_merged['SignupMonth']).apply(lambda x: x.n)
    
    cohort_data = df_merged.groupby(['SignupMonth', 'CohortIndex'])['customer_id'].nunique().reset_index()
    cohort_pivot = cohort_data.pivot(index='SignupMonth', columns='CohortIndex', values='customer_id')
    
    cohort_size = cohort_pivot.iloc[:, 0]
    retention = (cohort_pivot.divide(cohort_size, axis=0).round(3) * 100).fillna(0).reset_index()
    
    # DB 저장 전처리 (타입 변환) 
    if 'created_at' in rfm.columns: rfm['created_at'] = rfm['created_at'].astype(str)
    if 'start_at' in rfm.columns: rfm['start_at'] = rfm['start_at'].astype(str)
    
    retention['SignupMonth'] = retention['SignupMonth'].astype(str)
    
    #  DB에 저장 (Snapshot)
    rfm.to_sql(name='rfm_snapshot', con=engine, if_exists='replace', index=False)
    retention.to_sql(name='cohort_snapshot', con=engine, if_exists='replace', index=False)
    
    print("✅ [배치 완료] RFM(LTV 포함) 및 코호트 분석 결과가 DB에 저장되었습니다.")


# API 엔드포인트
@app.post("/api/analysis/trigger-batch")
async def trigger_analysis(background_tasks: BackgroundTasks):
    """(1) 분석 실행 트리거: 백그라운드에서 분석 후 DB 저장"""
    background_tasks.add_task(perform_full_analysis_and_save)
    return {"status": "started", "message": "배치 분석이 시작되었습니다. 잠시 후 DB를 확인하세요."}

@app.get("/api/analysis/db-check")
def check_rfm_ltv_data():
    """(2) RFM 및 LTV 결과 조회 (DB에서 읽기)"""
    try:
        # 안전하게 NaN을 0으로 한 번 더 처리
        df = pd.read_sql("SELECT customer_id, segment, total_score, LTV FROM rfm_snapshot LIMIT 5", con=engine)
        return {"rfm_ltv_sample": df.fillna(0).to_dict(orient='records')}
    except Exception as e:
        return {"message": "데이터가 없습니다. 배치를 먼저 실행하세요.", "error": str(e)}

@app.get("/api/analysis/cohort-check")
def check_cohort_data():
    """(3) 코호트 분석 결과 조회 (DB에서 읽기)"""
    try:
        df = pd.read_sql("SELECT * FROM cohort_snapshot", con=engine)
        # 안전하게 NaN을 0으로 한 번 더 처리
        return {"cohort_data": df.fillna(0).to_dict(orient='records')}
    except Exception as e:
        return {"message": "코호트 데이터가 없습니다. 배치를 먼저 실행하세요.", "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)