from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import numpy as np
from .database import ojo_engine, analysis_engine
from .analyzer.ltv_analyzer import calculate_ltv
from .analyzer.cohort_analyzer import calculate_segmented_cohort
from .analyzer.advice_analyzer import get_member_advice_timeline
from .analyzer.subscription_analyzer import calculate_subscription
from .analyzer.regional_sales_analyzer import calculate_regional_sales
from .analyzer.churn_prediction_analyzer import calculate_churn_prediction
from .model.recommendation import recommendation_engine

app = FastAPI(title="High-5 Data Science Server")

origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
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
    print("[분석] LTV(고객 생애 가치) 계산 중...")
    ltv_df = calculate_ltv(ojo_engine)
    if not ltv_df.empty:
        ltv_df.to_sql('ltv_snapshot', con=analysis_engine, if_exists='replace', index=False)

    # 2. 코호트 세그먼트 리스트 정의 및 계산
    segments = ['all', 'high_consult', 'vip', 'big_spender']
    all_cohort_results = []

    for seg in segments:
        try:
            print(f"[분석] {seg} 기준 코호트 분석 중...")
            df = calculate_segmented_cohort(ojo_engine, segment_type=seg)
            if not df.empty:
                all_cohort_results.append(df)
            else:
                print(f"[정보] {seg} 조건에 맞는 데이터가 없습니다.")
        except Exception as e:
            print(f"{seg} 코호트 분석 중 에러 발생: {e}")

    if all_cohort_results:
        final_cohort_df = pd.concat(all_cohort_results, ignore_index=True)
        final_cohort_df.to_sql('cohort_snapshot', con=analysis_engine, if_exists='replace', index=False)
        print(f"총 {len(segments)}개 세그먼트 코호트 적재 완료")

    # 4. 요금제별 이탈률 스냅샷 저장
    print("요금제별 이탈 통계 분석 시작...")
    sub_result = calculate_subscription(ojo_engine)

    if isinstance(sub_result, dict):
        if not sub_result['conversions'].empty:
            sub_result['conversions'].to_sql(
                'conversion_snapshot',
                con=analysis_engine,
                if_exists='replace',
                index=False
            )
        if not sub_result['product_churn'].empty:
            sub_result['product_churn'].to_sql(
                'churn_snapshot',
                con=analysis_engine,
                if_exists='replace',
                index=False
            )
        if not sub_result['top_reasons'].empty:
            sub_result['top_reasons'].to_sql(
                'reason_snapshot',
                con=analysis_engine,
                if_exists='replace',
                index=False
            )

    # 5. 이탈 예측 스냅샷 저장
    print("이탈 예측 분석 시작...")
    try:
        churn_result = calculate_churn_prediction(ojo_engine)

        if not churn_result["detail"].empty:
            churn_result["detail"].to_sql(
                'churn_prediction_snapshot',
                con=analysis_engine,
                if_exists='replace',
                index=False
            )

        if not churn_result["summary"].empty:
            churn_result["summary"].to_sql(
                'churn_prediction_summary_snapshot',
                con=analysis_engine,
                if_exists='replace',
                index=False
            )

        print("이탈 예측 스냅샷 적재 완료")
    except Exception as e:
        print(f"이탈 예측 분석 중 에러 발생: {e}")

    # 6. 지역별 분석 결과 스냅샷 저장
    print("지역별 분석 시작...")
    try:
        region_stats = calculate_regional_sales(ojo_engine, analysis_engine)
        if region_stats:
            region_df = pd.DataFrame(region_stats)
            region_df.to_sql('region_snapshot', con=analysis_engine, if_exists='replace', index=False)
            print("지역별 분석 스냅샷 적재 완료")
    except Exception as e:
        print(f"지역별 분석 중 에러 발생: {e}")

    print("분석 결과 적재 완료 (ojo_analysis)")

    # 맞춤 추천
    try:
        print("[AI 추천] 고객별 맞춤 상품 추천 계산 시작...")
        recommendation_engine(ojo_engine, analysis_engine) 
        print("[AI 추천] 추천 결과 스냅샷(recommend_snapshot) 적재 완료")
    except Exception as e:
        print(f"[AI 추천] 추천 엔진 실행 중 에러 발생: {e}")

@app.get("/api/analysis/make")
async def make_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis_pipeline)

    return {
        "status": "started",
        "message": "다차원 분석 및 스냅샷 적재를 백그라운드에서 안전하게 시작합니다."
    }


# 조회 API
@app.get("/api/analysis/ltv/{memberId}")
def get_member_ltv(memberId: str):
    df = pd.read_sql(f"SELECT * FROM ltv_snapshot WHERE member_id = :memberId", con=analysis_engine, params={"memberId": memberId})
    if not df.empty:
        return {"status": "success", "data": df.to_dict(orient='records')[0]}
    else:
        return {"status": "error", "message": f"{memberId} ltv 조회 실패", "data": {}}

@app.get("/api/analysis/cohort")
def get_cohort(segment: str = 'all'):
    query = f"SELECT * FROM cohort_snapshot WHERE segment_type = '{segment}'"
    df = pd.read_sql(query, con=analysis_engine)

    if df.empty:
        return {"status": "error", "message": f"No data found for segment: {segment}"}

    df = df.replace([np.inf, -np.inf], np.nan)
    result = df.to_dict(orient='records')
    clean_result = [{k: (None if pd.isna(v) else v) for k, v in record.items()} for record in result]

    return {"status": "success", "segment": segment, "data": clean_result}

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

# 요금제 통계
@app.get("/api/analysis/churn")
async def get_subscription():
    try:
        conversions = pd.read_sql("SELECT * FROM conversion_snapshot", con=analysis_engine)
        churn = pd.read_sql("SELECT * FROM churn_snapshot", con=analysis_engine)
        reasons = pd.read_sql("SELECT * FROM reason_snapshot", con=analysis_engine)

        return {
            "status": "SUCCESS",
            "data": {
                "conversions": conversions.to_dict(orient='records'),
                "product_churn": churn.to_dict(orient='records'),
                "top_reasons": reasons.to_dict(orient='records')
            }
        }
    except Exception as e:
        return {"status": "ERROR", "message": f"데이터가 아직 준비되지 않았습니다: {str(e)}"}


# 지역 통계
@app.get("/api/analysis/region")
async def get_regional_sales():
    try:
        df = pd.read_sql("SELECT * FROM region_snapshot", con=analysis_engine)
        return {"status": "SUCCESS", "data": df.to_dict(orient='records')}
    except Exception:
        return {"status": "ERROR", "message": "데이터를 불러올 수 없습니다."}

# 이탈률 예측
@app.get("/api/predictions/churn")
async def get_churn_prediction():
    try:
        df = pd.read_sql(
            "SELECT * FROM churn_prediction_summary_snapshot",
            con=analysis_engine
        )

        total_count = int(df["count"].sum()) if not df.empty else 0

        return {
            "status": "success",
            "data": {
                "totalAnalyzed": total_count,
                "riskDistribution": df.to_dict(orient="records") if not df.empty else []
            },
            "message": None
        }
    except Exception as e:
        return {
            "status": "error",
            "data": None,
            "message": "fail"
        }
   
# 맞춤 상품 추천
@app.get("/api/analysis/recommend/{memberId}")
def get_member_recommendation(memberId: int):
    """특정 고객의 맞춤형 추천 상품 Top 3 조회"""
    try:
        query = f"SELECT * FROM recommend_snapshot WHERE member_id = {memberId} ORDER BY rank ASC"
        df = pd.read_sql(query, con=analysis_engine)
        
        if df.empty:
            return {"status": "success", "data": [], "message": "추천 데이터가 없습니다."}
            
        return {
            "status": "success",
            "data": df.to_dict(orient='records')
        }
    except Exception as e:
        return {"status": "error", "message": str(e)} 

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)