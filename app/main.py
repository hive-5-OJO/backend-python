from fastapi import FastAPI, BackgroundTasks, Path
import pandas as pd
from database import ojo_engine, analysis_engine
from analyzer.ltv_analyzer import calculate_ltv
from analyzer.cohort_analyzer import calculate_cohort
from analyzer.subscription_analyzer import calculate_subscription

app = FastAPI(title="High-5 Data Science Server")

# ==========================================
# [분석 실행 로직] Spring이 호출함
# ==========================================
def run_analysis_pipeline():
    # 1. 고도화 분석 수행
    ltv_df = calculate_ltv(ojo_engine)
    cohort_df = calculate_cohort(ojo_engine)
    
    # 2. Spring이 계산한 RFM/KPI 데이터도 필요시 복사하거나 그대로 활용
    # 여기서는 파이썬이 분석한 결과만 ojo_analysis에 저장
    if not ltv_df.empty:
        ltv_df.to_sql('ltv_snapshot', con=analysis_engine, if_exists='replace', index=False)
    if not cohort_df.empty:
        cohort_df.to_sql('cohort_snapshot', con=analysis_engine, if_exists='replace', index=False)
    
    # 요금제별 이탈률 스냅샷 저장    
    sub_result = calculate_subscription(ojo_engine)
    if not sub_result['product_churn'].empty:
        sub_result['product_churn'].to_sql('churn_snapshot', con=analysis_engine, if_exists='replace', index=False)    
    
    print("✅ 분석 결과 적재 완료 (ojo_analysis)")

@app.get("/api/analysis/make")
async def make_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis_pipeline)
    return {"status": "started", "message": "LTV 및 코호트 분석을 시작합니다."}

# ==========================================
# [조회 API] 대시보드에서 호출함
# ==========================================
@app.get("/api/analysis/ltv/{memberId}")
def get_member_ltv(memberId: str):
    df = pd.read_sql(f"SELECT * FROM ltv_snapshot WHERE member_id = '{memberId}'", con=analysis_engine)
    return {"status": "success", "data": df.to_dict(orient='records')[0] if not df.empty else {}}

@app.get("/api/analysis/cohort")
def get_cohort():
    df = pd.read_sql("SELECT * FROM cohort_snapshot", con=analysis_engine)
    return {"status": "success", "data": df.to_dict(orient='records')}

# Spring이 ojo DB에 넣어둔 KPI를 그대로 조회하는 API (Proxy 역할)
@app.get("/api/analysis/dashboard")
def get_dashboard():
    # Spring이 계산해서 ojo DB에 넣어둔 KPI 테이블들을 읽어옴
    summary = pd.read_sql("SELECT * FROM kpi_summary_metrics", con=ojo_engine)
    return {"status": "success", "data": summary.to_dict(orient='records')}

@app.get("/api/analysis/churn")
async def get_subscription():
    result = calculate_subscription(ojo_engine)
    
    # 날짜 포맷팅 (Series가 비어있지 않을 때만 수행)
    if not result['conversions'].empty:
        result['conversions']['start_month'] = result['conversions']['start_month'].astype(str)
    
    return {
        "status": "SUCCESS",
        "data": {
            "conversions": result['conversions'].to_dict(orient='records'),
            "product_churn": result['product_churn'].to_dict(orient='records'),
            "top_reasons": result['top_reasons'].to_dict(orient='records')
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)