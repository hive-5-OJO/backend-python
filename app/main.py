from fastapi import FastAPI, BackgroundTasks, Path
import pandas as pd
import sys
import io

# 1. 분리해둔 모듈들 불러오기
from database import engine
from data_loader import load_data_from_csv
from analyzer import analyze_all

# 윈도우 터미널 한글 깨짐 방지 설정
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding='utf-8')

app = FastAPI(title="High-5 Data Analysis Batch Server (API Spec Integrated)")

# ==========================================
# [함수] 배치 파이프라인 컨트롤러
# ==========================================
def run_batch_process(base_date: str):
    """ 데이터 로드 -> 분석 -> DB 저장의 전체 파이프라인 조율 """
    
    # 1단계: 데이터 로드
    dfs = load_data_from_csv()
    if not dfs:
        print("[오류] 데이터 로딩 실패로 배치를 중단합니다.")
        return
        
    # 2단계: 핵심 분석 엔진 가동 (analyzer.py)
    print(f"[분석 진행] 기준일({base_date})로 데이터 분석을 시작합니다...")
    results = analyze_all(dfs, base_date)

    # 3단계: 분석 결과 DB 저장
    print("[DB 저장] 분석 완료! 생성된 모든 테이블을 DB에 밀어 넣습니다...")
    for table_name, df in results.items():
        df.to_sql(name=table_name, con=engine, if_exists='replace', index=False)
        
    print("[배치 완료] 데이터 분석 및 DB 저장이 완벽하게 종료되었습니다.")

# ==========================================
# [API 엔드포인트] - 팀 API 명세서 완벽 반영
# ==========================================

# 1. analysis 테이블 생성 (rfm, ltv 등 계산)
# API 명세서에 GET /api/analysis/make 로 정의되어 있으므로 GET 파라미터로 받습니다.
@app.get("/api/analysis/make")
async def make_analysis_tables(background_tasks: BackgroundTasks, base_date: str = "2027-01-01"):
    background_tasks.add_task(run_batch_process, base_date)
    return {"status": "started", "message": f"기준일({base_date})로 분석 테이블(RFM, LTV, 통계 등) 생성을 시작합니다."}

# 2. LTV 결과 조회 (단일 고객)
@app.get("/api/analysis/ltv/{memberId}")
def get_member_ltv(memberId: str = Path(..., description="조회할 고객의 ID")):
    try:
        query = f"SELECT member_id, exact_score, rfm_rank, top_percent, LTV FROM rfm_snapshot WHERE member_id = '{memberId}'"
        df = pd.read_sql(query, con=engine)
        if df.empty:
            return {"status": "fail", "message": f"'{memberId}' 고객의 데이터를 찾을 수 없습니다."}
        return {"status": "success", "data": df.fillna(0).to_dict(orient='records')[0]}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 3. 코호트 분석 조회
@app.get("/api/analysis/cohort")
def get_cohort_data():
    try:
        df = pd.read_sql("SELECT * FROM cohort_snapshot", con=engine)
        return {"status": "success", "data": df.fillna(0).to_dict(orient='records')}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 4. KPI + 세그먼트 (대시보드 종합)
@app.get("/api/analysis/dashboard")
def get_dashboard_data():
    try:
        summary = pd.read_sql("SELECT * FROM kpi_summary_metrics", con=engine).fillna(0).to_dict(orient='records')
        top_plans = pd.read_sql("SELECT * FROM kpi_top_plans", con=engine).fillna(0).to_dict(orient='records')
        top_cs = pd.read_sql("SELECT * FROM kpi_top_cs", con=engine).fillna(0).to_dict(orient='records')
        subs_status = pd.read_sql("SELECT * FROM kpi_subs_status", con=engine).fillna(0).to_dict(orient='records')
        
        return {
            "status": "success",
            "data": {
                "summary_metrics": summary,
                "top_plans": top_plans,
                "top_cs": top_cs,
                "subscription_status": subs_status
            }
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 5. 상담 카테고리별 비중 조회
@app.get("/api/advice/categories")
def get_advice_categories():
    try:
        df = pd.read_sql("SELECT * FROM kpi_top_cs", con=engine)
        return {"status": "success", "data": df.fillna(0).to_dict(orient='records')}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 6. 상담 시간대별 통계
@app.get("/api/advice/time")
def get_advice_time():
    try:
        df = pd.read_sql("SELECT * FROM kpi_advice_time", con=engine)
        return {"status": "success", "data": df.fillna(0).to_dict(orient='records')}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 7. 상담 만족도 통계 조회
@app.get("/api/advice/satisfaction")
def get_advice_satisfaction():
    try:
        df = pd.read_sql("SELECT * FROM kpi_advice_sat", con=engine)
        return {"status": "success", "data": df.fillna(0).to_dict(orient='records')}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 8. 상담사별 성과 조회
@app.get("/api/analysis/admin")
def get_admin_performance():
    try:
        df = pd.read_sql("SELECT * FROM kpi_admin_perf", con=engine)
        return {"status": "success", "data": df.fillna(0).to_dict(orient='records')}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 9. 지역별 고객 분포 및 매출 통계 조회
@app.get("/api/analysis/region")
def get_region_stats():
    try:
        df = pd.read_sql("SELECT * FROM kpi_region_stats", con=engine)
        return {"status": "success", "data": df.fillna(0).to_dict(orient='records')}
    except Exception as e:
        return {"status": "error", "message": str(e)}

# 10. 고객별 상담 이력 조회
@app.get("/api/advice/{memberId}")
def get_advice_history(memberId: str = Path(...)):
    # ※ DB에 실제 존재하는 테이블 이름으로 변경 필요 (예: advice)
    query = f"SELECT * FROM advice_202601 WHERE member_id = '{memberId}'" 
    try:
        df = pd.read_sql(query, con=engine)
        return {"status": "success", "data": df.fillna(0).to_dict(orient='records')}
    except Exception as e:
        return {"status": "error", "message": "해당 고객의 상담 이력이 없습니다."}

# 11. 고객별 상담 타임라인
@app.get("/api/advice/timeline/{memberId}")
def get_advice_timeline(memberId: str = Path(...)):
    # ※ DB에 실제 존재하는 테이블 이름으로 변경 필요 (예: advice)
    query = f"SELECT start_at, category_id, satisfaction_score FROM advice_202601 WHERE member_id = '{memberId}' ORDER BY start_at DESC"
    try:
        df = pd.read_sql(query, con=engine)
        return {"status": "success", "data": df.fillna(0).to_dict(orient='records')}
    except Exception as e:
        return {"status": "error", "message": "해당 고객의 타임라인이 없습니다."}
    
@app.get("/api/advice/outbound")
def get_advice_outbound():
    try:
        df = pd.read_sql("SELECT * FROM kpi_advice_outbound", con=engine)
        return {"status": "success", "data": df.fillna(0).to_dict(orient='records')}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)