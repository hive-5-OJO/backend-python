# CREATE VIEW v_member_ai_features AS
# SELECT 
#     m.member_id, m.gender, m.birth_date, m.region, m.household_type,
#     fu.total_usage_amount, fu.usage_active_days_30d, fu.usage_peak_hour, -- 사용 패턴
#     fm.total_revenue, fm.avg_monthly_bill, fm.payment_delay_count, -- 결제 패턴
#     fc.top_consult_category, fc.total_complaint_count, -- 상담 패턴
#     cp.churn_score, cp.risk_grade, -- 이탈 예측
#     an.rfm_score, an.type as rfm_segment, an.ltv, -- 가치 분석
#     sp.product_id as current_product_id -- 현재 사용 중인 상품
# FROM member m
# LEFT JOIN feature_usage fu ON m.member_id = fu.member_id
# LEFT JOIN feature_monetary fm ON m.member_id = fm.member_id
# LEFT JOIN feature_consultation fc ON m.member_id = fc.member_id
# LEFT JOIN churn_prediction cp ON m.member_id = cp.member_id
# LEFT JOIN analysis an ON m.member_id = an.member_id
# LEFT JOIN subscription_period sp ON m.member_id = sp.member_id AND sp.status = 'ACTIVE';

import pandas as pd

def recommendation_engine():
    members = pd.read_sql("SELECT * FROM v_member_ai_features", db_conn)
    products = pd.read_sql("SELECT * FROM product", db_conn)
    recommendations = []

    for _, member in members.iterrows():
        # 사용 중인 상품 제외
        candidates = products[products['product_id'] != member['current_product_id']].copy()
        candidates['score'] = 0.0
        candidates['reason'] = ""

        if member['total_usage_amount'] > 0: 
            # 데이터 사용량 많으면 무제한 요금제 추천 - 뭐가 무제한인지 확인
            idx = candidates['product_name']
            candidates.loc[idx, 'score'] += 40
            candidates.loc[idx, 'reason'] += "데이터 사용량이 많아 무제한 요금제를 추천합니다."

        if member['risk_grade'] in ['DANGER', 'WARNING']:
            idx = candidates['product_category']    

    save_to_db(recommendations)