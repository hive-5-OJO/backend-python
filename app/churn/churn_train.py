import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier

from app.churn.churn_data_loader import (
    load_feature_consultation,
    load_feature_monetary,
    load_feature_lifecycle,
    load_feature_usage,
    load_member,
    load_main_subscription_period
)
from app.churn.churn_label_maker import make_churn_label
from app.churn.churn_preprocess import (
    build_base_dataset,
    add_derived_features,
    build_preprocessor,
    get_feature_columns
)
from app.churn.churn_evaluate import evaluate_model


ARTIFACT_DIR = "app/churn/artifacts"


def get_models():
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "HistGradientBoostingClassifier": HistGradientBoostingClassifier(
            max_iter=200,
            learning_rate=0.05,
            max_depth=6,
            random_state=42
        )
    }


def main():
    os.makedirs(ARTIFACT_DIR, exist_ok=True)

    print("1. 데이터 로딩 시작", flush=True)

    consultation = load_feature_consultation()
    monetary = load_feature_monetary()
    lifecycle = load_feature_lifecycle()
    usage = load_feature_usage()
    member = load_member()
    subscription = load_main_subscription_period()

    print("1. 데이터 로딩 시작", flush=True)
    df = build_base_dataset(consultation, monetary, lifecycle, usage, member)
 
    print("3. 파생 컬럼 생성", flush=True)
    df = add_derived_features(df)

    print("4. y(label) 생성 - 60일 기준", flush=True)
    df = make_churn_label(df, subscription, churn_days=60)

    print("\n[y 분포]", flush=True)
    print(df["y"].value_counts(dropna=False))
    print(df["y"].value_counts(normalize=True, dropna=False))

    numeric_features, categorical_features, binary_features = get_feature_columns()
    feature_cols = numeric_features + categorical_features + binary_features

    X = df[feature_cols]
    y = df["y"]

    print("\n5. train / test split", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=42
    )

    preprocessor = build_preprocessor()
    models = get_models()

    results = []
    trained_pipelines = {}  # 학습된 파이프라인을 모델명별로 보관
    best_model_name = None
    best_score = -1
    best_pipeline = None

    print("\n6. 모델 학습 및 평가", flush=True)
    for model_name, model in models.items():
        print(f"\n[학습 중] {model_name}")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        pipeline.fit(X_train, y_train)
        trained_pipelines[model_name] = pipeline  #  운영 저장용으로 보관

        y_pred = pipeline.predict(X_test)

        if not hasattr(pipeline, "predict_proba"):
            raise ValueError(f"{model_name} does not support predict_proba")

        y_proba = pipeline.predict_proba(X_test)[:, 1]

        metrics = evaluate_model(y_test, y_pred, y_proba)
        metrics["model"] = model_name
        results.append(metrics)

        print(metrics)

        if metrics["roc_auc"] > best_score:
            best_score = metrics["roc_auc"]
            best_model_name = model_name
            best_pipeline = pipeline

    result_df = pd.DataFrame(results).sort_values(by="roc_auc", ascending=False)

    print("\n[최종 성능 비교]")
    print(result_df[["model", "accuracy", "precision", "recall", "f1", "roc_auc"]])

    result_df.to_csv(f"{ARTIFACT_DIR}/model_comparison.csv", index=False)

    print(f"\nBest model by evaluation: {best_model_name}, ROC-AUC={best_score}")

    # 운영용 저장 모델은 HistGradientBoostingClassifier로 명확히 고정
    deployment_model_name = "HistGradientBoostingClassifier"
    deployment_pipeline = trained_pipelines[deployment_model_name]

    print(f"Deployment model: {deployment_model_name}")

    joblib.dump(deployment_pipeline, f"{ARTIFACT_DIR}/best_model.pkl")
    print(f"Best model saved -> {ARTIFACT_DIR}/best_model.pkl")


if __name__ == "__main__":
    main()