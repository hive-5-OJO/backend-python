from fastapi import FastAPI, BackgroundTasks, Path
import pandas as pd
import numpy as np
from .database import ojo_engine, analysis_engine
from .analyzer.ltv_analyzer import calculate_ltv
from .analyzer.cohort_analyzer import calculate_cohort

app = FastAPI(title="High-5 Data Science Server")

# [분석 실행 로직] Spring이 호출함
def run_analysis_pipeline():
    ltv_df = calculate_ltv(ojo_engine)
    cohort_df = calculate_cohort(ojo_engine)
    
    if not ltv_df.empty:
        ltv_df.to_sql('ltv_snapshot', con=analysis_engine, if_exists='replace', index=False)
    if not cohort_df.empty:
        cohort_df.to_sql('cohort_snapshot', con=analysis_engine, if_exists='replace', index=False)
    
    print("분석 결과 적재 완료 (ojo_analysis)")

@app.get("/api/analysis/make")
async def make_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis_pipeline)
    return {"status": "started", "message": "LTV 및 코호트 분석을 시작합니다."}

# 조회 API
@app.get("/api/analysis/ltv/{memberId}")
def get_member_ltv(memberId: str):
    df = pd.read_sql(f"SELECT * FROM ltv_snapshot WHERE member_id = '{memberId}'", con=analysis_engine)
    return {"status": "success", "data": df.to_dict(orient='records')[0] if not df.empty else {}}

@app.get("/api/analysis/cohort")
def get_cohort():
    # 1. DB에서 데이터를 읽어옵니다.
    df = pd.read_sql("SELECT * FROM cohort_snapshot", con=analysis_engine)
    
    df = df.replace([np.inf, -np.inf], np.nan)
    
    result = df.to_dict(orient='records')
    
    clean_result = [
        {k: (None if pd.isna(v) else v) for k, v in record.items()}
        for record in result
    ]
    
    return {"status": "success", "data": clean_result}

@app.get("/api/analysis/dashboard")
def get_dashboard():
    summary = pd.read_sql("SELECT * FROM rfm_kpi", con=ojo_engine)
    return {"status": "success", "data": summary.to_dict(orient='records')}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)