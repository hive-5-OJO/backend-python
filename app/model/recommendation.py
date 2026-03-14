import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from datetime import datetime

# 1GB = 1000
DATA_LIMIT_MAP = {
    # 5G 프리미어가 4개 
    # 매일 5GB 기준 한달 환산
    '5G 시그니처': 9999999, '5G 프리미어': 9999999, '너겟 59': 9999999, '너겟 65': 9999999, '너겟 69': 9999999,
    '너겟 26': 6000,'너겟 30': 7000,'너겟 33': 12000,'너겟 42': 36000,'너겟 45': 50000,
    '너겟 46': 80000,'너겟 47': 100000,'너겟 51': 1500000,
    '5G 스탠다드': 150000, '유쓰 5G 스탠다드': 210000, # 유쓰가 일반보다 혜택이 큼
    '5G 스탠다드 에센셜': 125000, '유쓰 5G 스탠다드 에센셜': 185000,
    '5G 데이터 슈퍼': 95000, '유쓰 5G 데이터 슈퍼': 145000,
    '5G 데이터 플러스': 80000, '유쓰 5G 데이터 플러스': 115000,
    '5G 데이터 레귤러': 50000, '유쓰 5G 데이터 레귤러': 60000,
    '5G 심플+': 31000, '유쓰 5G 심플+': 41000,
    '5G 베이직+': 24000, '유쓰 5G 베이직+': 30000,
    '5G 라이트+': 14000, '유쓰 5G 슬림+': 10000,
    '5G 슬림+': 9000, '5G 라이트 청소년': 8000,
    '5G 시니어 A': 10000, '5G 시니어 B': 9000, '5G 시니어 C': 8000,
    '(LTE) 데이터 시니어 33': 1700,
    '(LTE) 추가 요금 걱정 없는 데이터 시니어 69': 155000,
    '시니어16.5': 500,
    '5G 미니': 5000, '너겟 51': 30000,
    '(LTE) 추가 요금 걱정 없는 데이터 69': 155000,
    '(LTE) 추가 요금 걱정 없는 데이터 청소년 33': 2000,
    'LTE 다이렉트 45': 150000, 
    'LTE 다이렉트 22': 1300, '(LTE) 데이터 33': 1500,
    '5G 키즈 45': 9000, '5G 키즈 39': 3300, 'LTE청소년19': 350
}

def get_recommendations(member_id, ojo_engine, analysis_engine):
    # 속도 향상을 위해 뷰 사용 가능
    query = """
    SELECT 
        m.member_id, m.gender, m.birth_date, m.region, m.household_type, m.created_at as join_date,
        fu.total_usage_amount, fu.usage_active_days_30d, fu.usage_peak_hour,
        fm.avg_monthly_bill, fm.total_revenue, fm.payment_delay_count,
        fc.top_consult_category, fc.total_complaint_count,
        an.rfm_score, an.type as segment, an.ltv,
        -- cp.churn_score, cp.risk_grade, 
        sp.product_id as current_product_id
    FROM member m
    JOIN feature_usage fu ON m.member_id = fu.member_id
    JOIN feature_monetary fm ON m.member_id = fm.member_id
    JOIN feature_consultation fc ON m.member_id = fc.member_id
    JOIN analysis an ON m.member_id = an.member_id
    -- LEFT JOIN churn_prediction cp ON m.member_id = cp.member_id
    LEFT JOIN subscription_period sp ON m.member_id = sp.member_id AND sp.status = 'ACTIVE'
    """
    df = pd.read_sql(query, con=ojo_engine)
    products = pd.read_sql("SELECT * FROM product", con=ojo_engine)

    features = ['household_type', 'total_usage_amount', 'usage_active_days_30d', 'avg_monthly_bill', 
                'usage_peak_hour', 'payment_delay_count', 'total_consult_count', 'age',
                'ltv', 'rfm_score',]
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features].fillna(0))

    user_sim = cosine_similarity(df_scaled)
    user_sim_df = pd.DataFrame(user_sim, index=df['member_id'], columns=df['member_id'])

    # 대상 고객 정보 추출
    target_user = df[df['member_id'] == member_id].iloc[0]
    target_data_limit = DATA_LIMIT_MAP.get(target_user['current_product_name'], 0)

    # 대상 고객과 가장 유사한 상위 10명 추출
    similar_users = user_sim_df[member_id].sort_values(ascending=False).iloc[1:11].index

    # 유사한 사용자들이 가장 많이 사용하는 상품 집계 (현재 내 상품 제외)
    # target_user_product = df[df['member_id'] == member_id]['current_product_id'].values[0]
    # 
    # recommended_products = df[df['member_id'].isin(similar_users)]
    # recommend_list = recommended_products[recommended_products['current_product_id'] != target_user_product]
    similar_user_products = df[df['member_id'].isin(similar_users)]['current_product_id'].unique()

    recommendations = []
    current_year = datetime.now().year
    age = current_year - pd.to_datetime(target_user['birth_date']).year 

    for _, p in products.iterrows():
        # 현재 사용자가 사용하고 있거나 사용하고 있는 거보다 데이터 한도 낮은거 빼기
        if p['product_name'] == target_user['current_product_name']: continue
        if DATA_LIMIT_MAP.get(p['product_name'], 0) < target_data_limit : continue
        
        # 유사도 : 비즈니스 = 3:7
        scores = 30 if p['product_id'] in similar_user_products else 0 
        reasons = np.array([""] * len(df), dtype=object)

        # 최대 데이터 사용량의 90% 이상 사용자에게 자신의 사용금액에 추가 2만원 이내의 초과 상품 추천하기
        usage_stress = (df['total_usage_amount'] / target_data_limit) > 0.9
        up_sell_mask = (p['price'] > df['avg_monthly_bill']) & (p['price'] <= df['avg_monthly_bill'] + 20000)
        scores += np.where(usage_stress & up_sell_mask, 25, 0)
        reasons = np.where(up_sell_mask & usage_stress, "데이터 사용 패턴에 맞춰 한 단계 높은 요금제를 추천합니다. ", reasons)

        # 가족 구성 3인 이상이면 결합 상품 추천 => db에 lg u+ 결합 상품 추가(U+투게더 결합, 참 쉬운 가족 결합 ,신혼플러스 결합, 참 쉬운 케이블 가족 결합)
        if any(k in p['product_name'] for k in ['가족', '결합', '투게더']):
            family_mask = (df['household_type'] >= 3)
            scores += np.where(family_mask, 20, 0)
            reasons = np.where(family_mask, "다인 가구 맞춤 결합 상품으로 통신비를 절감해보세요. ", reasons)

        # 연령
        if '키즈' in p['product_name']:
            mask = (age <= 12)
            scores = np.where(mask, scores + 20, -999)
            reasons = np.where(mask, f"어린이 고객님(만 {age}세)을 위한 맞춤 요금제입니다.", reasons)
        elif '청소년' in p['product_name']:
            mask = (age > 12) & (age <= 18)
            scores = np.where(mask, scores + 20, -999)
            reasons = np.where(mask, f"청소년 고객님(만 {age}세)을 위한 맞춤 요금제입니다.", reasons)
        elif '유쓰' in p['product_name']:
            mask = (age > 18) & (age <= 34)
            scores = np.where(mask, scores + 20, -999)
            reasons = np.where(mask, f"청년 고객님(만 {age}세)을 위한 맞춤 요금제입니다.", reasons)
        elif '시니어' in p['product_name']:
            mask = (age >= 65) 
            scores = np.where(mask, scores + 20, -999) 
            reasons = np.where(mask, f"실버 고객님(만 {age}세)을 위한 맞춤 요금제입니다. ", reasons)   

        # 가입 기간 - 15점
        if pd.notnull(df['join_date']):
            join_date = pd.to_datetime(df['join_date'])
            calculation_base_date = datetime(datetime.now().year, 11, 30)
            tenure_years = (calculation_base_date - join_date).days // 365
            
            # 장기 고객 (2년, 5년, 10년 이상) - 결합 할인 및 VIP 혜택 강조
            # 장기 고객 전용 혜택 필요(현재 존재하지 않는 거 같음)
            # 실제 로직은 약간 변경 필요
            if tenure_years >= 10:
                print("")
            elif tenure_years >= 5:
                print("")
            elif tenure_years >= 2:
                print("")

        # 세그먼트 - LOST, RISK, COMMON , LOYAL, vip 
        if p['product_category'] == 'ADDON_SERVICE':
            vip_mask = (df['rfm_type'] == 'VIP')
            scores += np.where(vip_mask, 30, 0)
            reasons = np.where(vip_mask, "VIP 고객님께 제공되는 전용 라이프스타일 혜택입니다. ", reasons)
            
            # 일반 고객에게는 '체험형' 부가서비스 추천 (저렴한 것 위주)
            gen_mask = (df['rfm_type'] != 'VIP') & (p['price'] < 5000)
            scores += np.where(gen_mask, 20, 0)
            reasons = np.where(gen_mask, "부담 없는 가격으로 이용 가능한 인기 부가서비스입니다. ", reasons)
        risk_high_ltv = (df['type'] == 'RISK') & (df['ltv'] > df['ltv'].quantile(0.8))
        discount_p = (p['price'] < df['current_base_price'])
        scores += np.where(risk_high_ltv & discount_p, 65, 0)
        reasons = np.where(risk_high_ltv & discount_p, "장기 우수 고객님을 위한 요금 다이어트 제안입니다. ", reasons)    

        # 연체
        if df['payment_delay_count'] >= 1:
            down_sell_target = (p['price'] < df['avg_monthly_bill']) & (p['product_category'] == 'BASE')
            scores += np.where((df['payment_delay_count'] > 0) & down_sell_target, 20, 0)    
            reasons = np.where(down_sell_target, "가계 통신비 절감을 위한 실속형 요금제입니다.", reasons)

        # 상담 카테고리
        if '로밍' in p['product_name']: 
            roaming_mask = (df['last_consult_category'] == 18) # 로밍 관련 ID
            scores += np.where(roaming_mask, 50, 0)
            reasons = np.where(roaming_mask, "최근 로밍 상담을 바탕으로 가장 인기 있는 로밍 상품을 추천합니다. ", reasons)
        if p['product_category'] == 'BASE' and p['price'] < 50000:
            churn_mask = (df['churn_score'] > 0.8) | (df['last_consult_category'] == 19) # 해지 상담
            scores += np.where(churn_mask, 40, 0)
            reasons = np.where(churn_mask, "고객님을 위한 특별 할인 요금제와 혜택을 확인해보세요. ", reasons)    

        if scores[target_user.name] >= 30:
            recommendations.append({
                'member_id': member_id,
                'recommended_product': p['product_name'],
                'price' : p['price'],
                'score': scores[target_user.name],
                'reason': reasons[target_user.name]
            })
  
    if not recommendations: return []
    final_res = pd.DataFrame(recommendations)

    final_res['rank'] = final_res.groupby('member_id')['score'].rank(ascending=False, method='first').astype(int)
    return final_res[final_res['rank'] <= 3].to_dict('records')
