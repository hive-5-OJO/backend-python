from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse 
import pandas as pd
import numpy as np
import io
from datetime import datetime
from urllib.parse import quote

# 데이터베이스 및 분석 모듈
from .database import ojo_engine, analysis_engine
from .analyzer.ltv_analyzer import calculate_ltv
from .analyzer.cohort_analyzer import calculate_segmented_cohort
from .analyzer.advice_analyzer import get_member_advice_timeline
from .analyzer.subscription_analyzer import calculate_subscription
from .analyzer.regional_sales_analyzer import calculate_regional_sales
from .analyzer.churn_prediction_analyzer import calculate_churn_prediction
from .model.recommendation import recommendation_engine
from .analyzer.rfm_analyzer import calculate_rfm_metrics

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

# 5. 통합 analysis 테이블 생성 (RFM 기반 자동화)
    print("[통합] 최종 analysis 테이블 생성 중...")
    try:
        # DB에서 RFM 기반 세그먼트 계산해오기
        rfm_metrics_df = calculate_rfm_metrics(ojo_engine)
        
        if not ltv_df.empty and not rfm_metrics_df.empty:
            from datetime import datetime
            
            ltv_df.columns = ltv_df.columns.str.lower()
            
            analysis_final_df = ltv_df.copy()
            
            analysis_final_df = pd.merge(analysis_final_df, rfm_metrics_df, on='member_id', how='left')
            
            # 빈칸 방어
            analysis_final_df['rfm_score'] = analysis_final_df['rfm_score'].fillna(0)
            analysis_final_df['type'] = analysis_final_df['type'].fillna('일반')
            analysis_final_df['lifecycle_stage'] = analysis_final_df['lifecycle_stage'].fillna('ACTIVE')
            analysis_final_df['created_at'] = datetime.now()

            # 테이블 적재
            analysis_final_df[[
                'member_id', 'ltv', 'rfm_score', 'type', 'lifecycle_stage', 'created_at'
            ]].to_sql('analysis', con=analysis_engine, if_exists='replace', index=True, index_label='analysis_id')
            
            print("analysis 통합 테이블 적재 성공! (RFM 모델 적용 완료)")
        else:
            print("기준이 되는 LTV나 RFM 데이터가 없어서 analysis 테이블을 만들지 못했습니다.")
    except Exception as e:
        print(f"통합 테이블 생성 중 에러: {e}")

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
# @app.get("/api/advice/timeline/{memberId}")
# def get_member_timeline(memberId: int):
#     try:
#         df = get_member_advice_timeline(ojo_engine, memberId)

#         return {
#             "status": "success",
#             "data": {
#                 "memberId": memberId,
#                 "timeline": df.to_dict(orient='records') if not df.empty else []
#             },
#             "message": None
#         }
#     except Exception as e:
#         return {"status": "error", "data": None, "message": str(e)}

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


    # 통합 고객 분석 데이터 조회 (LTV, RFM, 등급, 생애주기)
@app.get("/api/analysis/customer/{memberId}")
def get_customer_analysis(memberId: int):
    try:
        # analysis 테이블에서 해당 고객의 가장 최근 데이터 1건 조회
        query = f"SELECT * FROM analysis WHERE member_id = {memberId} ORDER BY created_at DESC LIMIT 1"
        df = pd.read_sql(query, con=analysis_engine)
        
        if df.empty:
            return {
                "status": "success", 
                "data": {}, 
                "message": "해당 고객의 분석 데이터가 아직 없습니다."
            }
            
        result = df.to_dict(orient='records')[0]
        
        import numpy as np
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