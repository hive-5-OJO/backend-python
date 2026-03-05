from fastapi import FastAPI, BackgroundTasks, Path
import pandas as pd
import numpy as np
from .database import ojo_engine, analysis_engine
from .analyzer.ltv_analyzer import calculate_ltv
from .analyzer.cohort_analyzer import calculate_segmented_cohort

app = FastAPI(title="High-5 Data Science Server")

# [분석 실행 로직] Spring이 호출함
def run_analysis_pipeline():
    print("다차원 분석 파이프라인 가동...")
    
    # 1. LTV 계산 및 저장
    ltv_df = calculate_ltv(ojo_engine)
    if not ltv_df.empty:
        ltv_df.to_sql('ltv_snapshot', con=analysis_engine, if_exists='replace', index=False)

    # 2. 코호트 세그먼트 리스트 정의
    segments = ['all', 'high_consult', 'vip', 'big_spender']
    all_cohort_results = []

    for seg in segments:
        try:
            df = calculate_segmented_cohort(ojo_engine, segment_type=seg)
            if not df.empty:
                all_cohort_results.append(df)
        except Exception as e:
            print(f"❌ {seg} 분석 중 에러 발생: {e}")

    # 3. 모든 결과를 하나의 테이블로 합쳐서 저장
    if all_cohort_results:
        final_cohort_df = pd.concat(all_cohort_results, ignore_index=True)
        final_cohort_df.to_sql('cohort_snapshot', con=analysis_engine, if_exists='replace', index=False)
        print(f"총 {len(segments)}개 세그먼트 코호트 적재 완료")

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
def get_cohort(segment: str = 'all'):
    query = f"SELECT * FROM cohort_snapshot WHERE segment_type = '{segment}'"
    df = pd.read_sql(query, con=analysis_engine)
    
    if df.empty:
        return {"status": "error", "message": f"No data found for segment: {segment}"}

    # JSON 에러 방지 처리 (기존 로직 유지)
    df = df.replace([np.inf, -np.inf], np.nan)
    result = df.to_dict(orient='records')
    clean_result = [{k: (None if pd.isna(v) else v) for k, v in record.items()} for record in result]
    
    return {"status": "success", "segment": segment, "data": clean_result}

@app.get("/api/analysis/dashboard")
def get_dashboard():
    summary = pd.read_sql("SELECT * FROM rfm_kpi", con=ojo_engine)
    return {"status": "success", "data": summary.to_dict(orient='records')}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)