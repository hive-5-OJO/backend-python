from fastapi import FastAPI, BackgroundTasks, Path
import pandas as pd
import numpy as np
from .database import ojo_engine, analysis_engine
from .analyzer.ltv_analyzer import calculate_ltv
from .analyzer.cohort_analyzer import calculate_segmented_cohort
from .analyzer.subscription_analyzer import calculate_subscription
from .analyzer.regional_sales_analyzer import calculate_regional_sales
from .analyzer.churn_prediction_analyzer import calculate_churn_prediction
from .analyzer.advice_analyzer import get_member_advice_timeline

app = FastAPI(title="High-5 Data Science Server")

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

    # 이탈 예측 스냅샷 저장
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


def clean_df(df):
    if df.empty:
        return []
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.where(pd.notnull(df), None).to_dict(orient='records')


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
    df = pd.read_sql(f"SELECT * FROM ltv_snapshot WHERE member_id = '{memberId}'", con=analysis_engine)
    return {"status": "success", "data": df.to_dict(orient='records')[0] if not df.empty else {}}


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


# 추후 삭제될 수도 있음
@app.get("/api/analysis/dashboard")
def get_dashboard():
    summary = pd.read_sql("SELECT * FROM rfm_kpi", con=ojo_engine)
    return {"status": "success", "data": summary.to_dict(orient='records')}


# 요금제 통계
@app.get("/api/analysis/churn")
async def get_subscription():
    result = calculate_subscription(ojo_engine)

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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)