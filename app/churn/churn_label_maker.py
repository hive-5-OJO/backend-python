import pandas as pd


def make_churn_label(
    base_df: pd.DataFrame,
    subscription_df: pd.DataFrame,
    churn_days: int = 60
) -> pd.DataFrame:
    """
    feature_base_date 이후 churn_days 이내에
    메인 상품의 실제 종료일(end_at)이 존재하면 y=1
    """

    df = base_df.copy()
    sub = subscription_df.copy()

    df["feature_base_date"] = pd.to_datetime(df["feature_base_date"], errors="coerce")
    sub["end_at"] = pd.to_datetime(sub["end_at"], errors="coerce")

    merged = df.merge(
        sub[["member_id", "end_at"]],
        on="member_id",
        how="left"
    )

    merged["is_churn_event"] = (
        merged["end_at"].notna()
        & (merged["end_at"] > merged["feature_base_date"])
        & (merged["end_at"] <= merged["feature_base_date"] + pd.to_timedelta(churn_days, unit="D"))
    ).astype(int)

    label_df = (
        merged.groupby(["member_id", "feature_base_date"], as_index=False)["is_churn_event"]
        .max()
        .rename(columns={"is_churn_event": "y"})
    )

    result = df.merge(label_df, on=["member_id", "feature_base_date"], how="left")
    result["y"] = result["y"].fillna(0).astype(int)

    return result