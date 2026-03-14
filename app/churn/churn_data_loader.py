import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()


def get_engine():
    db_user = os.environ["DB_USER"]
    db_password = os.environ["DB_PASSWORD"]
    db_host = os.environ["DB_HOST"]
    db_port = os.environ["DB_PORT"]
    db_name = os.environ["DB_NAME"]

    url = f"mysql+pymysql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return create_engine(url)


def load_feature_consultation() -> pd.DataFrame:
    query = """
    SELECT
        member_id,
        feature_base_date,
        total_consult_count,
        last_7d_consult_count,
        last_30d_consult_count,
        avg_monthly_consult_count,
        last_consult_date,
        night_consult_count,
        weekend_consult_count,
        top_consult_category,
        total_complaint_count,
        last_consult_days_ago
    FROM feature_consultation
    """
    return pd.read_sql(query, get_engine())


def load_feature_monetary() -> pd.DataFrame:
    query = """
    SELECT
        member_id,
        feature_base_date,
        total_revenue,
        last_payment_amount,
        avg_monthly_bill,
        last_payment_date,
        payment_count_6m,
        monthly_revenue,
        payment_delay_count,
        prev_monthly_revenue,
        is_vip_prev_month,
        avg_order_val,
        purchase_cycle
    FROM feature_monetary
    """
    return pd.read_sql(query, get_engine())


def load_feature_lifecycle() -> pd.DataFrame:
    query = """
    SELECT
        member_id,
        feature_base_date,
        member_lifetime_days,
        days_since_last_activity,
        contract_end_days_left,
        is_dormant_flag,
        is_new_customer_flag,
        is_terminated_flag,
        signup_date
    FROM feature_lifecycle
    """
    return pd.read_sql(query, get_engine())


def load_feature_usage() -> pd.DataFrame:
    query = """
    SELECT
        member_id,
        feature_base_date,
        total_usage_amount,
        avg_daily_usage,
        max_usage_amount,
        usage_peak_hour,
        premium_service_count,
        last_activity_date,
        usage_active_days_30d
    FROM feature_usage
    """
    return pd.read_sql(query, get_engine())


def load_member() -> pd.DataFrame:
    query = """
    SELECT
        member_id,
        gender,
        birth_date,
        region,
        household_type,
        status,
        created_at
    FROM member
    """
    return pd.read_sql(query, get_engine())


def load_main_subscription_period() -> pd.DataFrame:
    query = """
    SELECT
        sp.member_id,
        sp.started_at,
        sp.end_at,
        sp.status AS subscription_status,
        p.product_id,
        p.product_name,
        p.product_type,
        p.product_category
    FROM subscription_period sp
    JOIN product p
      ON sp.product_id = p.product_id
    WHERE p.product_category = 'BASE'
    """
    return pd.read_sql(query, get_engine())