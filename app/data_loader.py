import pandas as pd
import os
import re

def find_single_csv(base_path, filename):
    for root, dirs, files in os.walk(base_path):
        if filename in files:
            return os.path.join(root, filename)
    return None

def find_and_concat_monthly_csvs(base_path, pattern):
    df_list = []
    found_files = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if re.match(pattern, file):
                file_path = os.path.join(root, file)
                try:
                    df_list.append(pd.read_csv(file_path))
                    found_files.append(file)
                except Exception as e:
                    print(f"[경고] {file} 읽기 실패: {e}")
                    
    if df_list:
        print(f" -> {len(found_files)}개의 파일을 찾아 병합했습니다.")
        return pd.concat(df_list, ignore_index=True)
    return None

def load_data_from_csv():
    """
    [임시 데이터 로더] 
    추후 스프링에서 DB에 1차 정제 데이터를 넣어주면, 이 함수는 통째로 삭제하거나 DB 조회용으로 변경합니다.
    """
    print("\n[데이터 로딩] 1년 치(12개월) CSV 파일을 자동 병합합니다...")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.abspath(os.path.join(current_dir, "..", ".."))
    
    dfs = {}
    dfs['df_id'] = find_and_concat_monthly_csvs(base_dir, r'^invoice_detail_2026\d{2}\.csv$')
    dfs['df_i'] = find_and_concat_monthly_csvs(base_dir, r'^invoice_2026\d{2}\.csv$')
    dfs['df_a'] = find_and_concat_monthly_csvs(base_dir, r'^advice_2026\d{2}\.csv$')
    dfs['df_p'] = find_and_concat_monthly_csvs(base_dir, r'^payment_2026\d{2}\.csv$')
    dfs['df_du'] = find_and_concat_monthly_csvs(base_dir, r'^data_usage_2026\d{2}\.csv$')

    single_files = {
        'df_c': 'categories.csv',
        'df_s': 'subscription_period.csv',
        'df_con': 'member_consent.csv',
        'df_m': 'member.csv'
    }
    
    for key, filename in single_files.items():
        found_path = find_single_csv(base_dir, filename)
        if found_path:
            dfs[key] = pd.read_csv(found_path)
        else:
            print(f"[경고] 단일 파일 누락: {filename}")
            return None

    for key, df in list(dfs.items()):
        if df is None or df.empty:
            if key in ['df_p', 'df_du']:
                dfs[key] = pd.DataFrame() 
            else:
                print(f"[오류] 필수 데이터({key}) 누락으로 중단합니다.")
                return None

    print("✅ 데이터 로딩 성공!\n")
    return dfs