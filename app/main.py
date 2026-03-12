from fastapi import FastAPI, BackgroundTasks, Path
import pandas as pd
import numpy as np
import io
from datetime import datetime
from fastapi.responses import StreamingResponse
from .database import ojo_engine, analysis_engine
from .analyzer.ltv_analyzer import calculate_ltv
from .analyzer.cohort_analyzer import calculate_segmented_cohort
from .analyzer.subscription_analyzer import calculate_subscription
from .analyzer.regional_sales_analyzer import calculate_regional_sales
from .analyzer.advice_analyzer import get_member_advice_timeline
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="High-5 Data Science Server")

# 프론트엔드 개발 서버 주소
origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:5173",
    "http://high5-ojo.s3-website.ap-northeast-2.amazonaws.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # GET, POST, OPTIONS 등 모든 메서드 허용
    allow_headers=["*"], # 모든 헤더 허용
)

# [분석 실행 로직] Spring이 호출함
def run_analysis_pipeline():
    print("다차원 분석 파이프라인 가동...")
    
    # 1. LTV 계산 및 저장
    ltv_df = calculate_ltv(ojo_engine)
    if not ltv_df.empty:
        ltv_df.to_sql('ltv_snapshot', con=analysis_engine, if_exists='replace', index=False)

    # 2. 코호트 세그먼트 리스트 정의 및 계산
    segments = ['all', 'high_consult', 'vip', 'big_spender']
    all_cohort_results = []

    for seg in segments:
        try:
            df = calculate_segmented_cohort(ojo_engine, segment_type=seg)
            if not df.empty:
                all_cohort_results.append(df)
        except Exception as e:
            print(f"{seg} 코호트 분석 중 에러 발생: {e}")

    # 3. 모든 결과를 하나의 테이블로 합쳐서 저장
    if all_cohort_results:
        final_cohort_df = pd.concat(all_cohort_results, ignore_index=True)
        final_cohort_df.to_sql('cohort_snapshot', con=analysis_engine, if_exists='replace', index=False)
        print(f"총 {len(segments)}개 세그먼트 코호트 적재 완료")

    # 요금제별 이탈률 스냅샷 저장    
    sub_result = calculate_subscription(ojo_engine)
    if not sub_result['product_churn'].empty:
        sub_result['product_churn'].to_sql('churn_snapshot', con=analysis_engine, if_exists='replace', index=False)    
    
    # 지역별 분석 결과 스냅샷 저장
    region_stats = calculate_regional_sales(ojo_engine)
    if region_stats: 
        region_df = pd.DataFrame(region_stats)
        region_df.to_sql('region_snapshot', con=analysis_engine, if_exists='replace', index=False)

    print("분석 결과 적재 완료 (ojo_analysis)")

def clean_df(df):
    if df.empty: return []
    # NaN은 None으로, Inf는 매우 큰 수나 None으로 교체
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.where(pd.notnull(df), None).to_dict(orient='records')

@app.get("/api/analysis/make")
async def make_analysis(background_tasks: BackgroundTasks):
    # 사용자가 API를 찌르면, 파이프라인 함수를 백그라운드 작업으로 던져놓습니다.
    background_tasks.add_task(run_analysis_pipeline)
    
    # 분석이 끝나길 기다리지 않고 바로 성공 응답을 줍니다! (타임아웃 방지)
    return {
        "status": "started", 
        "message": "다차원 분석 및 스냅샷 적재를 백그라운드에서 안전하게 시작합니다."
    }

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

# 추후 삭제될 수도 있음
@app.get("/api/analysis/dashboard")
def get_dashboard():
    summary = pd.read_sql("SELECT * FROM rfm_kpi", con=ojo_engine)
    return {"status": "success", "data": summary.to_dict(orient='records')}

# 요금제 통계
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
# 지역 통계
@app.get("/api/analysis/region")
async def get_regional_sales():
    try:
        df = pd.read_sql("SELECT * FROM region_snapshot", con=analysis_engine)
        return {"status": "SUCCESS", "data": clean_df(df)}
    # 테이블 없다면
    except Exception:
        return {"status": "SUCCESS", "data": []}
    
# 고객별 상담 타임라인
@app.get("/api/advice/timeline/{memberId}")
def get_member_timeline(memberId: int):
    try:
        df = get_member_advice_timeline(ojo_engine, memberId)
        
        return {
            "status": "success", 
            "data": {
                "memberId": memberId,
                "timeline": df.to_dict(orient='records') if not df.empty else []
            },
            "message": None
        }
    except Exception as e:
        return {"status": "error", "data": None, "message": str(e)}


@app.get("/api/analysis/report/export")
def export_analysis_report():
    try:
        #메모리 버퍼 생성 (가상의 엑셀 파일)
        output = io.BytesIO()
        
        #Pandas ExcelWriter를 사용해 여러 시트(Tab) 작성
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            
            df_cohort = pd.read_sql("SELECT * FROM cohort_snapshot", con=analysis_engine)
            if not df_cohort.empty:
                df_cohort.to_excel(writer, sheet_name="코호트_분석", index=False)

            df_region = pd.read_sql("SELECT * FROM region_snapshot", con=analysis_engine)
            if not df_region.empty:
                df_region.to_excel(writer, sheet_name="지역별_매출", index=False)

            df_ltv = pd.read_sql("SELECT * FROM ltv_snapshot", con=analysis_engine)
            if not df_ltv.empty:
                df_ltv.to_excel(writer, sheet_name="LTV_분석", index=False)

            df_churn = pd.read_sql("SELECT * FROM churn_snapshot", con=analysis_engine)
            if not df_churn.empty:
                df_churn.to_excel(writer, sheet_name="요금제_이탈률", index=False)

        output.seek(0)
        
        #다운로드 파일명 생성 (예: analysis_report_20260311.xlsx)
        today_str = datetime.now().strftime("%Y%m%d")
        filename = f"analysis_report_{today_str}.xlsx"
        
        #브라우저가 파일 다운로드로 인식하도록 헤더 설정
        headers = {
            'Content-Disposition': f'attachment; filename="{filename}"'
        }
        
        #스트리밍 방식으로 엑셀 파일 전송
        return StreamingResponse(
            output, 
            headers=headers, 
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )
        
    except Exception as e:
        return {"status": "error", "message": f"보고서 생성 중 오류 발생: {str(e)}"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)