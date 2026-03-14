import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_base_dataset(
    consultation: pd.DataFrame,
    monetary: pd.DataFrame,
    lifecycle: pd.DataFrame,
    usage: pd.DataFrame,
    member: pd.DataFrame
) -> pd.DataFrame:
    # consultation 기준으로 시작
    df = consultation.copy()

    # feature_base_date는 consultation 것을 대표로 사용
    df = df.merge(
        monetary.drop(columns=["feature_base_date"], errors="ignore"),
        on="member_id",
        how="inner"
    )
    df = df.merge(
        lifecycle.drop(columns=["feature_base_date"], errors="ignore"),
        on="member_id",
        how="inner"
    )
    df = df.merge(
        usage.drop(columns=["feature_base_date"], errors="ignore"),
        on="member_id",
        how="inner"
    )
    df = df.merge(member, on="member_id", how="left")

    return df


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    # 날짜 파싱
    result["feature_base_date"] = pd.to_datetime(result["feature_base_date"], errors="coerce")
    result["birth_date"] = pd.to_datetime(result["birth_date"], errors="coerce")
    result["created_at"] = pd.to_datetime(result["created_at"], errors="coerce")
    result["signup_date"] = pd.to_datetime(result["signup_date"], errors="coerce")

    # age 파생
    ref_date = result["feature_base_date"].dt.normalize()
    result["age"] = ((ref_date - result["birth_date"]).dt.days / 365.25).fillna(0).astype(int)

    # Y/N -> 1/0
    yn_cols = ["is_vip_prev_month", "is_dormant_flag", "is_new_customer_flag"]
    for col in yn_cols:
        if col in result.columns:
            result[col] = result[col].map({"Y": 1, "N": 0}).fillna(0).astype(int)

    return result


def get_feature_columns():
    numeric_features = [
        "total_consult_count",
        "last_7d_consult_count",
        "last_30d_consult_count",
        "avg_monthly_consult_count",
        "night_consult_count",
        "weekend_consult_count",
        "total_complaint_count",
        "last_consult_days_ago",
        "total_revenue",
        "last_payment_amount",
        "avg_monthly_bill",
        "payment_count_6m",
        "monthly_revenue",
        "payment_delay_count",
        "prev_monthly_revenue",
        "avg_order_val",
        "purchase_cycle",
        "member_lifetime_days",
        "days_since_last_activity",
        "contract_end_days_left",
        "total_usage_amount",
        "avg_daily_usage",
        "max_usage_amount",
        "usage_peak_hour",
        "premium_service_count",
        "usage_active_days_30d",
        "household_type",
        "age"
    ]

    categorical_features = [
        "top_consult_category",
        "gender",
        "region"
    ]

    binary_features = [
        "is_vip_prev_month",
        "is_dormant_flag",
        "is_new_customer_flag"
    ]

    return numeric_features, categorical_features, binary_features


def build_preprocessor():
    numeric_features, categorical_features, binary_features = get_feature_columns()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    binary_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
            ("bin", binary_transformer, binary_features),
        ]
    )
    return preprocessor