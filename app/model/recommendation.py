# CREATE VIEW v_member_ai_features AS
# SELECT 
#     m.member_id, m.gender, m.birth_date, m.region, m.household_type,
#     fu.total_usage_amount, fu.usage_active_days_30d, fu.usage_peak_hour, -- 사용 패턴
#     fm.total_revenue, fm.avg_monthly_bill, fm.payment_delay_count, -- 결제 패턴
#     fc.top_consult_category, fc.total_complaint_count, -- 상담 패턴
#     cp.churn_score, cp.risk_grade, -- 이탈 예측
#     an.rfm_score, an.type as rfm_type, an.ltv, -- 가치 분석
#     sp.product_id as current_product_id -- 현재 사용 중인 상품
# FROM member m
# LEFT JOIN feature_usage fu ON m.member_id = fu.member_id
# LEFT JOIN feature_monetary fm ON m.member_id = fm.member_id
# LEFT JOIN feature_consultation fc ON m.member_id = fc.member_id
# LEFT JOIN churn_prediction cp ON m.member_id = cp.member_id
# LEFT JOIN analysis an ON m.member_id = an.member_id
# LEFT JOIN subscription_period sp ON m.member_id = sp.member_id AND sp.status = 'ACTIVE';

import pandas as pd
from datetime import datetime

def recommendation_engine(ojo_engine, analysis_engine):
    members = pd.read_sql("SELECT * FROM v_member_ai_features", con=ojo_engine)
    products = pd.read_sql("SELECT * FROM product", con=ojo_engine)
    offered_promotions = pd.read_sql("SELECT member_id, promotion_id FROM advice WHERE promotion_id IS NOT NULL", con=ojo_engine)
    promotions = pd.read_sql("SELECT * FROM promotion", con=ojo_engine)
    recommendations = []

    current_year = datetime.now().year

    for _, member in members.iterrows():
        # 사용 중인 상품 제외
        candidates = products[products['product_id'] != member['current_product_id']].copy()
        candidates['score'] = 0.0
        candidates['reason'] = ""

        # 데이터 사용량에 대한 추천
        if member['total_usage_amount'] > 20000: 
            # 데이터 사용량 많으면 무제한 요금제 추천 - 뭐가 무제한인지 확인
            # 실제 자신의 요금제에서 지원하는 데이터 사용량까지만 사용할 확률이 크기에 이에 대한 로직 수정 필요
            idx = candidates['price'] >= 95000
            candidates.loc[idx, 'score'] += 40
            candidates.loc[idx, 'reason'] += "데이터 사용량이 많아 무제한 요금제를 추천합니다."

        # 이탈 위험 고객을 위한 추천
        if member['risk_grade'] in ['DANGER', 'WARNING']:
            used_promos = offered_promotions[offered_promotions['member_id'] == member['member_id']]['promotion_id'].tolist()
            new_promos = promotions[~promotions['promotion_id'].isin(used_promos)]
            
            if not new_promos.empty:
                idx = candidates['product_category'] == 'BASE' 
                candidates.loc[idx, 'score'] += 60
                candidates.loc[idx, 'reason'] += "이탈 위험 고객님을 위한 혜택 상품입니다.\n"

        # 나이에 따른 추천
        if pd.notnull(member['birth_date']):
            age = current_year - pd.to_datetime(member['birth_date']).year 
            # 다시 한 번 확인 필요
            if age <= 12:
                idx = candidates['product_name'].str.contains('키즈') 
                candidates.loc[idx, 'score'] += 50
                candidates.loc[idx, 'reason'] += f"어린이 고객님(만 {age}세)을 위한 맞춤 요금제입니다.\n" 
            elif age <= 18:
                idx = candidates['product_name'].str.contains('청소년') 
                candidates.loc[idx, 'score'] += 50
                candidates.loc[idx, 'reason'] += f"청소년 고객님(만 {age}세)을 위한 맞춤 요금제입니다.\n"
            elif age <= 34:
                idx = candidates['product_name'].str.contains('유쓰') 
                candidates.loc[idx, 'score'] += 50
                candidates.loc[idx, 'reason'] += f"청년 고객님(만 {age}세)을 위한 맞춤 요금제입니다.\n"          
            elif age >= 65:
                idx = candidates['product_name'].str.contains('시니어') 
                candidates.loc[idx, 'score'] += 50
                candidates.loc[idx, 'reason'] += f"실버 고객님(만 {age}세)을 위한 맞춤 요금제입니다.\n"  

        # rmm, ltv를 통한 추천(vip)
        if member['rfm_type'] == 'VIP' or member['ltv'] > 1000000: #ltv 값 조정 필요
            # 진짜 관련되었는지 다른 요소와 함께 해서 확인 필요 
            idx = candidates['product_category'] == 'ADDON_SERVICE'
            candidates.loc[idx, 'score'] += 30
            candidates.loc[idx, 'reason'] += "VIP 고객님께 어울리는 프리미엄 부가서비스를 추천합니다.\n"

        # 가성비 추천 - 빼도 됨
        if member['avg_monthly_bill'] and member['avg_monthly_bill'] > 80000:
            idx = (candidates['price'] < member['avg_monthly_bill']) & (candidates['product_category'] == 'BASE')
            candidates.loc[idx, 'score'] += 20
            candidates.loc[idx, 'reason'] += "현재 요금보다 저렴하면서 혜택은 비슷한 요금제입니다.\n"

        # 결제 패턴
        if member['payment_delay_count'] >= 1:
            # 연체 이력이 있으면 저렴한 요금제 추천
            idx = (candidates['price'] < member['avg_monthly_bill']) & (candidates['product_category'] == 'BASE')
            candidates.loc[idx, 'score'] += 45
            candidates.loc[idx, 'reason'] += "가계 통신비 절감을 위한 실속형 요금제입니다.\n"

        # 가족 구성원으로 인한 추천
        if member['household_type'] >= 3:
            idx = candidates['product_name'].str.contains('가족|결합|투게더') # 실제로는 U+투게더 결합, 참 쉬운 가족 결합 ,신혼플러스 결합, 참 쉬운 케이블 가족 결합
            candidates.loc[idx, 'score'] += 30
            candidates.loc[idx, 'reason'] += "다인 가구 고객님을 위한 가족 결합 할인 상품입니다.\n"
        # 지역, 잠재 vip, 상담 이력, rfm에 따라 낮은 값 올리기 위한 추천

        # top3 제공
        top_candidates = candidates[candidates['score'] > 0].sort_values(by='score', ascending=False).head(3)
        
        for i, (_, best) in enumerate(top_candidates.iterrows()):
            recommendations.append({
                'member_id': int(member['member_id']),
                'recommended_product_id': int(best['product_id']),
                'recommended_product_name': best['product_name'],
                'reason': best['reason'].strip(),
                'score': float(best['score']),
                'rank': i + 1 
            })

    result_df = pd.DataFrame(recommendations)
    result_df.to_sql('recommend_snapshot', con=analysis_engine, if_exists='replace', index=False)