import joblib
import pandas as pd
from pathlib import Path

from app.churn.churn_data_loader import (
    load_feature_consultation,
    load_feature_monetary,
    load_feature_lifecycle,
    load_feature_usage,
    load_member,
)
from app.churn.churn_preprocess import (
    build_base_dataset,
    add_derived_features,
    get_feature_columns,
)

# 모델 파일 경로
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "churn" / "artifacts" / "best_model.pkl"


def _make_risk_grade(churn_score: float) -> str:
    """
    churn_score(0~1)를 등급으로 변환 (기준값 조정 가능)
    """
    if churn_score >= 0.7:
        return "DANGER"
    elif churn_score >= 0.4:
        return "WARNING"
    return "SAFE"


def _load_prediction_base_df() -> pd.DataFrame:
    """
    feature 4개 + member를 조인해서 예측용 입력 데이터 생성
    """
    consultation = load_feature_consultation()
    monetary = load_feature_monetary()
    lifecycle = load_feature_lifecycle()
    usage = load_feature_usage()
    member = load_member()

    df = build_base_dataset(
        consultation=consultation,
        monetary=monetary,
        lifecycle=lifecycle,
        usage=usage,
        member=member,
    )
    df = add_derived_features(df)

    return df


def calculate_churn_prediction(ojo_engine=None):
    """
    학습된 best_model.pkl 기반으로 churn score 예측
    반환 형식:
    {
        "detail": DataFrame(member_id, churn_score, risk_grade),
        "summary": DataFrame(grade, count, ratio)
    }
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"학습된 모델 파일이 없습니다: {MODEL_PATH}\n"
            f"먼저 `python -m app.churn.churn_train` 실행해서 best_model.pkl을 생성하세요."
        )

    # 1. 모델 로드
    model = joblib.load(MODEL_PATH)

    # 2. 예측용 데이터 로드
    df = _load_prediction_base_df()

    if df.empty:
        empty_detail = pd.DataFrame(columns=["member_id", "churn_score", "risk_grade"])
        empty_summary = pd.DataFrame(columns=["grade", "count", "ratio"])
        return {
            "detail": empty_detail,
            "summary": empty_summary
        }

    # 3. feature 컬럼 선택
    numeric_features, categorical_features, binary_features = get_feature_columns()
    feature_cols = numeric_features + categorical_features + binary_features

    missing_cols = [col for col in feature_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"예측에 필요한 컬럼이 없습니다: {missing_cols}")

    X = df[feature_cols].copy()

    # 4. 모델 예측
    if not hasattr(model, "predict_proba"):
        raise ValueError("현재 저장된 모델은 predict_proba를 지원하지 않습니다.")

    churn_scores = model.predict_proba(X)[:, 1]

    # 5. 결과 생성
    result_df = pd.DataFrame({
        "member_id": df["member_id"].values,
        "churn_score": churn_scores
    })

    result_df["risk_grade"] = result_df["churn_score"].apply(_make_risk_grade)

    # 소수점 정리
    result_df["churn_score"] = result_df["churn_score"].round(4)

    # 6. 상세 결과
    detail_df = result_df[["member_id", "churn_score", "risk_grade"]].copy()

    # 7. 요약 결과
    total = len(detail_df)
    summary_rows = []

    for grade in ["DANGER", "WARNING", "SAFE"]:
        count = int((detail_df["risk_grade"] == grade).sum())
        ratio = round((count / total) * 100, 2) if total > 0 else 0.0
        summary_rows.append({
            "grade": grade,
            "count": count,
            "ratio": ratio
        })

    summary_df = pd.DataFrame(summary_rows)

    return {
        "detail": detail_df,
        "summary": summary_df
    }
