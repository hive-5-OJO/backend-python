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

def get_all_recommendations(ojo_engine, analysis_engine):
    # 속도 향상을 위해 뷰 사용 가능
    query = """
    SELECT 
        m.member_id, m.gender, m.birth_date, m.region, m.household_type, m.created_at as join_date,
        fu.total_usage_amount, fu.usage_active_days_30d, fu.usage_peak_hour,
        fm.avg_monthly_bill, fm.total_revenue, fm.payment_delay_count,
        fc.top_consult_category, fc.total_consult_count,
        -- an.rfm_score, an.type as segment, an.ltv,
        -- cp.churn_score, cp.risk_grade, 
        sp.product_id as current_product_id,
        p.product_name as current_product_name, p.price as current_base_price
    FROM member m
    JOIN feature_usage fu ON m.member_id = fu.member_id
    JOIN feature_monetary fm ON m.member_id = fm.member_id
    JOIN feature_consultation fc ON m.member_id = fc.member_id
    -- JOIN analysis an ON m.member_id = an.member_id
    -- LEFT JOIN churn_prediction cp ON m.member_id = cp.member_id
    LEFT JOIN subscription_period sp ON m.member_id = sp.member_id AND sp.status = 'ACTIVE'
    LEFT JOIN product p ON sp.product_id = p.product_id
    """
    df = pd.read_sql(query, con=ojo_engine)
    products = pd.read_sql("SELECT * FROM product", con=ojo_engine)

    current_year = datetime.now().year
    df['age'] = current_year - pd.to_datetime(df['birth_date']).dt.year 
    df['current_limit'] = df['current_product_name'].map(DATA_LIMIT_MAP).fillna(0)

    features = ['household_type', 'total_usage_amount', 'usage_active_days_30d', 'avg_monthly_bill', 
                'usage_peak_hour', 'payment_delay_count', 'total_consult_count', 'age',]
                # 'ltv', 'rfm_score',]
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features].fillna(0))

    product_centroids = []
    for p_id in products['product_id']:
        centroid = df_scaled[df['current_product_id'] == p_id].mean(axis=0)
        product_centroids.append(centroid if not np.isnan(centroid).any() else np.zeros(len(features)))

    product_centroids = np.array(product_centroids)
    user_sim_matrix = cosine_similarity(df_scaled, product_centroids)

    num_users = len(df)
    num_products = len(products)

    # 유사도 : 비즈니스 = 3:7
    total_scores = user_sim_matrix * 30 
    total_reasons = np.full((num_users, num_products), "", dtype=object)

    p_names = products['product_name'].values
    p_prices = products['price'].values
    p_categories = products['product_category'].values
    p_limits = np.array([DATA_LIMIT_MAP.get(name, 0) for name in p_names])
    
    # 현재 사용자가 사용하고 있거나 사용하고 있는 거보다 데이터 한도 낮은 상품 제외
    is_current = df['current_product_name'].values[:, None] == p_names[None, :]
    is_lower = df['current_limit'].values[:, None] > p_limits[None, :]
    total_scores[is_current | is_lower] = -999

    # 최대 데이터 사용량의 90% 이상 + 가격 2만원 이내 상승 
    usage_stress = (df['total_usage_amount'] / df['current_limit'].replace(0, np.inf)).values > 0.9
    price_diff_mask = (p_prices[None, :] > df['avg_monthly_bill'].values[:, None]) & \
                      (p_prices[None, :] <= df['avg_monthly_bill'].values[:, None] + 20000)
    up_sell_mask = usage_stress[:, None] & price_diff_mask
    total_scores += np.where(up_sell_mask, 25, 0)
    total_reasons[up_sell_mask] += "데이터 사용 패턴에 맞춰 한 단계 높은 요금제를 추천합니다. "

    # 3인 이상 => db에 lg u+ 결합 상품 추가(U+투게더 결합, 참 쉬운 가족 결합 ,신혼플러스 결합, 참 쉬운 케이블 가족 결합)
    is_family_prod = np.array([any(k in name for k in ['가족', '결합', '투게더']) for name in p_names])
    family_user_mask = (df['household_type'] >= 3).values
    family_mask = family_user_mask[:, None] & is_family_prod[None, :]
    total_scores += np.where(family_mask, 20, 0)
    total_reasons[family_mask] += "다인 가구 맞춤 결합 상품으로 통신비를 절감해보세요. "

    # 연령
    age_masks = {
        '키즈': (df['age'] <= 12).values,
        '청소년': ((df['age'] > 12) & (df['age'] <= 18)).values,
        '유쓰': ((df['age'] > 18) & (df['age'] <= 34)).values,
        '시니어': (df['age'] >= 65).values
    }
    for keyword, user_mask in age_masks.items():
        prod_mask = np.array([keyword in name for name in p_names])
        total_scores += np.where(user_mask[:, None] & prod_mask[None, :], 20, 0)
        total_scores[~user_mask[:, None] & prod_mask[None, :]] = -999
        total_reasons[user_mask[:, None] & prod_mask[None, :]] += f"{keyword} 고객님을 위한 맞춤 요금제입니다. "

    # 가입 기간 - 15점
    join_date = pd.to_datetime(df['join_date'])
    calculation_base_date = datetime(datetime.now().year, 11, 30)
    tenure_years = (calculation_base_date - join_date).dt.days // 365

    tenure_10_mask = (tenure_years >= 10)
    tenure_5_mask = (tenure_years >= 5) & (tenure_years < 10)
    tenure_2_mask = (tenure_years >= 2) & (tenure_years < 5)

    total_scores += np.where(tenure_10_mask | tenure_5_mask | tenure_2_mask, 15, 0)
    total_reasons[tenure_5_mask] += "장기 우수 고객님을 위한 전용 혜택입니다. "
    
    # 세그먼트 - LOST, RISK, COMMON , LOYAL, vip 
    is_vip = (df['rfm_type'] == 'VIP').values
    is_addon_prod = (p_categories == 'ADDON_SERVICE')

    vip_addon_mask = is_vip[:, None] & is_addon_prod[None, :]
    total_scores += np.where(vip_addon_mask, 30, 0)
    total_reasons[vip_addon_mask] += "VIP 고객님을 위한 전용 혜택입니다. "
    # 일반 고객에게는 저렴한 체험형 부가서비스 추천
    is_gen_user = ~is_vip
    is_cheap_addon = is_addon_prod & (p_prices < 5000)
    gen_addon_mask = is_gen_user[:, None] & is_cheap_addon[None, :]
    total_scores += np.where(gen_addon_mask, 20, 0)
    total_reasons[gen_addon_mask] += "부담 없는 가격으로 이용 가능한 인기 부가서비스입니다. "
    # 위험 고객
    is_risk = (df['type'] == 'RISK').values
    is_high_ltv = (df['ltv'] > df['ltv'].quantile(0.8)).values
    is_cheaper_prod = p_prices[None, :] < df['current_base_price'].values[:, None]
    is_base_prod = (p_categories == 'BASE')

    diet_mask = is_risk[:, None] & is_high_ltv[:, None] & is_cheaper_prod & is_base_prod[None, :]
    total_scores += np.where(diet_mask, 30, 0)
    total_reasons[diet_mask] += "장기 우수 고객님을 위한 요금 다이어트 제안입니다. "
    
    # 연체
    has_delay = (df['payment_delay_count'] >= 1).values
    is_lower_bill = p_prices[None, :] < df['avg_monthly_bill'].values[:, None]

    delay_recom_mask = has_delay[:, None] & is_lower_bill & is_base_prod[None, :]
    total_scores += np.where(delay_recom_mask, 20, 0)
    total_reasons[delay_recom_mask] += "가계 통신비 절감을 위한 실속형 요금제입니다. " 

    # 상담 카테고리 => db에 상품 추가
    roaming_user_mask = (df['top_consult_category'] == 18).values # 로밍 관련 ID
    roaming_prod_mask = np.array(['로밍' in name for name in p_names])
    roaming_mask = roaming_user_mask[:, None] & roaming_prod_mask[None, :]
    total_scores += np.where(roaming_mask, 50, 0)
    total_reasons[roaming_mask] += "최근 로밍 상담을 바탕으로 인기 상품을 추천합니다. "

    # is_churn_consult = (df['top_consult_category'] == 19).values # 해지 상담
    # is_high_churn_score = (df['churn_score'] > 0.8).values
    # is_retention_target = is_churn_consult | is_high_churn_score

    # retention_mask = is_retention_target[:, None] & (p_prices[None, :] < 50000) & is_base_prod[None, :]
    # total_scores += np.where(retention_mask, 40, 0)
    # total_reasons[retention_mask] += "고객님을 위한 특별 할인 요금제와 혜택을 확인해보세요. "

    user_indices, prod_indices = np.where(total_scores >= 30)
    if len(user_indices) == 0: return "추천 조건에 맞는 항목이 없습니다."
    
    final_res = pd.DataFrame({
        'member_id': df['member_id'].values[user_indices],
        'product_name': p_names[prod_indices],
        'price': p_prices[prod_indices],
        'score': total_scores[user_indices, prod_indices],
        'reason': total_reasons[user_indices, prod_indices]
    })

    final_res['rank'] = final_res.groupby('member_id')['score'].rank(ascending=False, method='first').astype(int)
    result_snapshot = final_res[final_res['rank'] <= 3]
    result_snapshot.to_sql('recommend_snapshot', con=analysis_engine, if_exists='replace', index=False)
    return f"{len(result_snapshot)}개 성공"

def get_recommendations(member_id, ojo_engine):
    query = """
    SELECT 
        m.member_id, m.gender, m.birth_date, m.region, m.household_type, m.created_at as join_date,
        fu.total_usage_amount, fu.usage_active_days_30d, fu.usage_peak_hour,
        fm.avg_monthly_bill, fm.total_revenue, fm.payment_delay_count,
        fc.top_consult_category, fc.total_consult_count,
        -- an.rfm_score, an.type as segment, an.ltv,
        -- cp.churn_score, cp.risk_grade, 
        sp.product_id as current_product_id,
        p.product_name as current_product_name
    FROM member m
    JOIN feature_usage fu ON m.member_id = fu.member_id
    JOIN feature_monetary fm ON m.member_id = fm.member_id
    JOIN feature_consultation fc ON m.member_id = fc.member_id
    -- JOIN analysis an ON m.member_id = an.member_id
    -- LEFT JOIN churn_prediction cp ON m.member_id = cp.member_id
    LEFT JOIN subscription_period sp ON m.member_id = sp.member_id AND sp.status = 'ACTIVE'
    LEFT JOIN product p ON sp.product_id = p.product_id
    """
    df = pd.read_sql(query, con=ojo_engine)
    if df[df['member_id'] == member_id].empty: return []
    products = pd.read_sql("SELECT * FROM product", con=ojo_engine)

    current_year = datetime.now().year
    df['age'] = current_year - pd.to_datetime(df['birth_date']).dt.year

    # 대상 고객 정보 추출
    target_user = df[df['member_id'] == member_id].iloc[0]
    target_data_limit = DATA_LIMIT_MAP.get(target_user['current_product_name'], 0)
    age = target_user['age']

    features = ['household_type', 'total_usage_amount', 'usage_active_days_30d', 'avg_monthly_bill', 
                'usage_peak_hour', 'payment_delay_count', 'total_consult_count', 'age',]
                # 'ltv', 'rfm_score',]
    scaler = StandardScaler()
    df_scaled = scaler.fit_transform(df[features].fillna(0))

    target_idx = df[df['member_id'] == member_id].index[0]
    user_sim = cosine_similarity(df_scaled[target_idx].reshape(1, -1), df_scaled)[0]
    user_sim_series = pd.Series(user_sim, index=df['member_id'])

    # 대상 고객과 가장 유사한 상위 10명 추출
    similar_users = user_sim_series.sort_values(ascending=False).iloc[1:11].index

    # 유사한 사용자들이 가장 많이 사용하는 상품 집계 (현재 내 상품 제외)
    # target_user_product = df[df['member_id'] == member_id]['current_product_id'].values[0]
    # 
    # recommended_products = df[df['member_id'].isin(similar_users)]
    # recommend_list = recommended_products[recommended_products['current_product_id'] != target_user_product]
    similar_user_products = df[df['member_id'].isin(similar_users)]['current_product_id'].unique()

    recommendations = []

    for _, p in products.iterrows():
        # 현재 사용자가 사용하고 있거나 사용하고 있는 거보다 데이터 한도 낮은거 빼기
        if p['product_name'] == target_user['current_product_name']: continue
        if DATA_LIMIT_MAP.get(p['product_name'], 0) < target_data_limit : continue
        
        # 유사도 : 비즈니스 = 3:7
        scores = 30 if p['product_id'] in similar_user_products else 0 
        reasons = []

        # 연령
        age_match = True
        if '키즈' in p['product_name']:
            if age <= 12:
                scores += 20    
                reasons.append(f"어린이 고객님(만 {age}세)을 위한 맞춤 요금제입니다.")
            else: age_match = False
        elif '청소년' in p['product_name']:
            if 12 < age <= 18:
                scores += 20
                reasons.append(f"청소년 고객님(만 {age}세)을 위한 맞춤 요금제입니다.")
            else: age_match = False
        elif '유쓰' in p['product_name']:
            if 18 < age <= 34:
                scores += 20
                reasons.append(f"청년 고객님(만 {age}세)을 위한 맞춤 요금제입니다.")
            else: age_match = False
        elif '시니어' in p['product_name']:
            if age >= 65:
                scores += 20
                reasons.append(f"실버 고객님(만 {age}세)을 위한 맞춤 요금제입니다.") 
            else: age_match = False
        if not age_match: continue

        # 최대 데이터 사용량의 90% 이상 사용자에게 자신의 사용금액에 추가 2만원 이내의 초과 상품 추천하기
        if target_data_limit > 0 and (target_user['total_usage_amount'] / target_data_limit) > 0.9:
            if target_user['avg_monthly_bill'] < p['price'] <= target_user['avg_monthly_bill'] + 20000:
                scores += 25
        reasons.append("데이터 사용 패턴에 맞춰 한 단계 높은 요금제를 추천합니다. ")

        # 가족 구성 3인 이상이면 결합 상품 추천 => db에 lg u+ 결합 상품 추가(U+투게더 결합, 참 쉬운 가족 결합 ,신혼플러스 결합, 참 쉬운 케이블 가족 결합)
        if any(k in p['product_name'] for k in ['가족', '결합', '투게더']):
            if target_user['household_type'] >= 3:
                scores += 20
                reasons.append("다인 가구 맞춤 결합 상품으로 통신비를 절감해보세요. ")

        # 가입 기간 - 15점
        join_date = pd.to_datetime(target_user['join_date'])
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
        # if p['product_category'] == 'ADDON_SERVICE':
        #     if target_user['rfm_type'] == 'VIP':
        #         scores += 30
        #         reasons.append("VIP 고객님께 제공되는 전용 라이프스타일 혜택입니다. ")
            
        #     # 일반 고객에게는 '체험형' 부가서비스 추천 (저렴한 것 위주)
        #     if (target_user['rfm_type'] != 'VIP') & (p['price'] < 5000):
        #         scores += 20, 0
        #         reasons.append("부담 없는 가격으로 이용 가능한 인기 부가서비스입니다. ")
        # if (target_user['type'] == 'RISK') & (target_user['ltv'] > df['ltv'].quantile(0.8)):
        #     if (p['price'] < target_user['current_base_price']):
        #         scores += 25
        #         reasons.append("장기 우수 고객님을 위한 요금 다이어트 제안입니다. ")    

        # 연체
        if target_user['payment_delay_count'] >= 1:
            if (p['price'] < target_user['avg_monthly_bill']) & (p['product_category'] == 'BASE'):
                scores += 20    
                reasons.append("가계 통신비 절감을 위한 실속형 요금제입니다.")

        # 상담 카테고리
        if '로밍' in p['product_name'] and target_user['last_consult_category'] == 18: # 로밍 관련 ID 
            scores += 50
            reasons.append("최근 로밍 상담을 바탕으로 가장 인기 있는 로밍 상품을 추천합니다. ")
        # if p['product_category'] == 'BASE' and p['price'] < 50000 and (target_user['churn_score'] > 0.8) | (target_user['last_consult_category'] == 19): # 해지 상담
        #     scores += 40
        #     reasons.append("고객님을 위한 특별 할인 요금제와 혜택을 확인해보세요. ")    

        if scores >= 30:
            recommendations.append({
                'member_id': member_id,
                'recommended_product': p['product_name'],
                'price' : p['price'],
                'score': scores,
                'reason': ", ".join(reasons) if reasons else "고객님을 위한 맞춤 상품이 없습니다"
            })
  
    if not recommendations: return []
    final_res = pd.DataFrame(recommendations)
    
    final_res = final_res.sort_values(by='score', ascending=False)
    final_res['rank'] = range(1, len(final_res) + 1)
    final_res['created_at'] = datetime.now().strftime('%Y-%m-%d')
    return final_res.head(3).to_dict(orient='records')
