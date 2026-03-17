from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse 
import pandas as pd
import numpy as np
import io
from urllib.parse import quote

# 데이터베이스 및 분석 모듈
import traceback
import time
from datetime import datetime

from .database import ojo_engine, analysis_engine
from .analyzer.ltv_analyzer import calculate_ltv
from .analyzer.cohort_analyzer import calculate_segmented_cohort
from .analyzer.advice_analyzer import get_member_advice_timeline
from .analyzer.subscription_analyzer import calculate_subscription
from .analyzer.regional_sales_analyzer import calculate_regional_sales
from .analyzer.churn_prediction_analyzer import calculate_churn_prediction
from .analyzer.rfm_analyzer import calculate_rfm_metrics
from .model.recommendation import get_recommendations, get_all_recommendations

app = FastAPI(title="High-5 Data Science Server")

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
    start_time = time.time() 
    print(f"다차원 분석 파이프라인 가동: {datetime.now()}")

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

    # 3. 요금제별 이탈률 스냅샷 저장
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

    # 4. 이탈 예측 스냅샷 저장
    print("[CHURN-ML] 이탈 예측 분석 시작...", flush=True)
    try:
        churn_result = calculate_churn_prediction(ojo_engine)

        print(f"[CHURN-ML] detail rows = {len(churn_result['detail'])}", flush=True)
        print(f"[CHURN-ML] summary rows = {len(churn_result['summary'])}", flush=True)

        if not churn_result["detail"].empty:
            churn_result["detail"].to_sql(
                'churn_prediction_snapshot',
                con=analysis_engine,
                if_exists='replace',
                index=False
            )
            print("[CHURN-ML] churn_prediction_snapshot 저장 완료", flush=True)

        if not churn_result["summary"].empty:
            churn_result["summary"].to_sql(
                'churn_prediction_summary_snapshot',
                con=analysis_engine,
                if_exists='replace',
                index=False
            )
            print("[CHURN-ML] churn_prediction_summary_snapshot 저장 완료", flush=True)

        print("[CHURN-ML] 이탈 예측 스냅샷 적재 완료", flush=True)

    except Exception as e:
        print(f"[CHURN-ML] 이탈 예측 분석 중 에러 발생: {e}", flush=True)
        traceback.print_exc()

    # 6. 통합 analysis 테이블 생성 (RFM 기반 자동화)
    print("[통합] 최종 analysis 테이블 생성 중...")
    try:
        rfm_metrics_df = calculate_rfm_metrics(ojo_engine)

        if not ltv_df.empty and not rfm_metrics_df.empty:
            ltv_df.columns = ltv_df.columns.str.lower()

            analysis_final_df = ltv_df.copy()
            analysis_final_df = pd.merge(
                analysis_final_df,
                rfm_metrics_df,
                on='member_id',
                how='left'
            )

            analysis_final_df['rfm_score'] = analysis_final_df['rfm_score'].fillna(0)
            analysis_final_df['type'] = analysis_final_df['type'].fillna('일반')
            analysis_final_df['lifecycle_stage'] = analysis_final_df['lifecycle_stage'].fillna('ACTIVE')
            analysis_final_df['created_at'] = datetime.now()

            analysis_final_df[[
                'member_id', 'ltv', 'rfm_score', 'type', 'lifecycle_stage', 'created_at'
            ]].to_sql(
                'analysis',
                con=analysis_engine,
                if_exists='replace',
                index=True,
                index_label='analysis_id'
            )

            print("analysis 통합 테이블 적재 성공! (RFM 모델 적용 완료)")

            # 마이그레이션 적용 필요
            analysis_final_df[[
                'member_id', 'ltv', 'rfm_score', 'type', 'lifecycle_stage', 'created_at'
            ]].to_sql(
                'analysis',
                con=ojo_engine,
                if_exists='replace',
                index=False
            )
            print("analysis 통합 테이블 적재 성공! (RFM 모델 적용 완료)")
        else:
            print("기준이 되는 LTV나 RFM 데이터가 없어서 analysis 테이블을 만들지 못했습니다.")
    except Exception as e:
        print(f"통합 테이블 생성 중 에러: {e}")
        
    # 5. 지역별 분석 결과 스냅샷 저장
    print("지역별 분석 시작...")
    try:
        region_stats = calculate_regional_sales(ojo_engine, analysis_engine)
        if region_stats:
            region_df = pd.DataFrame(region_stats)
            region_df.to_sql('region_snapshot', con=analysis_engine, if_exists='replace', index=False)
            print("지역별 분석 스냅샷 적재 완료")
    except Exception as e:
        print(f"지역별 분석 중 에러 발생: {e}")


    # 7. 맞춤 추천
    try:
        print("[AI 추천] 고객별 맞춤 상품 추천 계산 시작...")
        get_all_recommendations(ojo_engine, analysis_engine) 
        print("[AI 추천] 추천 결과 스냅샷(recommend_snapshot) 적재 완료")
    except Exception as e:
        print(f"[AI 추천] 추천 엔진 실행 중 에러 발생: {e}")

    end_time = time.time()
    duration = end_time - start_time
    print("분석 결과 적재 완료 (ojo_analysis)")
    print(f"TOTAL EXECUTION TIME: {duration:.2f} seconds")

    print("분석 결과 적재 완료 (ojo_analysis)")

def clean_df(df):
    if df.empty: return []
    # NaN은 None으로, Inf는 매우 큰 수나 None으로 교체
    df = df.replace([np.inf, -np.inf], np.nan)
    return df.where(pd.notnull(df), None).to_dict(orient='records')

@app.get("/api/analysis/make")
async def make_analysis(background_tasks: BackgroundTasks):
    background_tasks.add_task(run_analysis_pipeline)

    return {
        "status": "started",
        "message": "다차원 분석 및 스냅샷 적재를 백그라운드에서 안전하게 시작합니다."
    }
# async def make_analysis(): 
#     run_analysis_pipeline() 
#     return {"status": "success", "message": "분석 완료"}


# 조회 API
@app.get("/api/analysis/ltv/{memberId}")
def get_member_ltv(memberId: str):
    df = pd.read_sql(
        "SELECT * FROM ltv_snapshot WHERE member_id = :memberId",
        con=analysis_engine,
        params={"memberId": memberId}
    )
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
        result = calculate_churn_prediction(ojo_engine)

        detail_df = result["detail"]
        summary_df = result["summary"]

        total_count = int(len(detail_df)) if not detail_df.empty else 0

        return {
            "status": "success",
            "data": {
                "totalAnalyzed": total_count,
                "riskDistribution": summary_df.to_dict(orient="records") if not summary_df.empty else [],
                "detail": detail_df.to_dict(orient="records") if not detail_df.empty else []
            },
            "message": None
        }

    except Exception as e:
        return {
            "status": "error",
            "data": None,
            "message": str(e)
        }


# 맞춤 상품 추천
@app.get("/api/analysis/recommend/{memberId}")
def get_member_recommendation(memberId: int):
    try:
        data = get_recommendations(memberId, ojo_engine)
        
        if not data:
            return {"status": "success", "data": [], "message": "추천 데이터가 없습니다."}

        return {
            "status": "success",
            "data": data
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}


# 통합 고객 분석 데이터 조회 (LTV, RFM, 등급, 생애주기)
@app.get("/api/analysis/customer/{memberId}")
def get_customer_analysis(memberId: int):
    try:
        query = f"SELECT * FROM analysis WHERE member_id = {memberId} ORDER BY created_at DESC LIMIT 1"
        df = pd.read_sql(query, con=analysis_engine)

        if df.empty:
            return {
                "status": "success",
                "data": {},
                "message": "해당 고객의 분석 데이터가 아직 없습니다."
            }

        result = df.to_dict(orient='records')[0]
        clean_result = {k: (None if pd.isna(v) else v) for k, v in result.items()}

        return {
            "status": "success",
            "data": clean_result,
            "message": "고객 통합 분석 데이터 조회 성공"
        }
    except Exception as e:
        return {
            "status": "error",
            "data": None,
            "message": f"분석 데이터 조회 중 오류 발생: {str(e)}"
        }
    
    # 엑셀 보고서 다운로드 API
@app.get("/api/analysis/report/export")
def export_analysis_report():
    print("[리포트] 엑셀 보고서 추출 시작...")
    try:
        # 1. DB에서 데이터 불러오기 (추천 테이블 제외, analysis만 조회)
        an_df = pd.read_sql("SELECT * FROM analysis", con=analysis_engine)

        if an_df.empty:
            return {"status": "error", "message": "추출할 분석 데이터가 아직 없습니다. 파이프라인을 먼저 실행해주세요."}

        # 2. 요약(Summary) 데이터 만들기 (추천 관련 지표 제거)
        summary_data = {
            "총 고객 수": [len(an_df)],
            "VIP 고객 수": [len(an_df[an_df['type'] == 'VIP'])],
            "이탈 위험(RISK) 고객 수": [len(an_df[an_df['type'] == 'RISK'])],
            "평균 LTV (예측 수익)": [int(an_df['ltv'].mean()) if not an_df.empty else 0],
            "보고서 생성일시": [datetime.now().strftime("%Y-%m-%d %H:%M:%S")]
        }
        summary_df = pd.DataFrame(summary_data)

        # 3. 메모리 상에서 엑셀 파일 만들기 (디스크 저장 X)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # 시트 1: 한눈에 보는 요약
            summary_df.to_excel(writer, sheet_name='핵심_요약', index=False)
            # 시트 2: 전체 고객 세그먼트 및 LTV
            an_df.to_excel(writer, sheet_name='고객_분석_상세', index=False)

        # 포인터를 처음으로 되돌리기 (중요!)
        output.seek(0)

        # 4. 파일명 한글 깨짐 방지 및 스트리밍 반환
        today_str = datetime.now().strftime("%Y%m%d")
        file_name = f"CRM_고객분석_보고서_{today_str}.xlsx"
        encoded_file_name = quote(file_name)

        headers = {
            'Content-Disposition': f'attachment; filename="{encoded_file_name}"'
        }

        # 엑셀 파일 스트리밍 리스폰스 반환
        return StreamingResponse(
            output, 
            headers=headers, 
            media_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        )

    except Exception as e:
        print(f"보고서 추출 중 에러: {e}")
        return {"status": "error", "message": f"보고서 생성 중 오류가 발생했습니다: {str(e)}"}



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
