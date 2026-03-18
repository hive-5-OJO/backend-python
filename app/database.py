import os
from dotenv import load_dotenv
from sqlalchemy import create_engine, QueuePool

load_dotenv()

# 환경 변수가 없으면 로컬 환경(127.0.0.1:3307)을 바라보도록 설정
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "3306") 

def get_url(db_name):
    return f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{DB_HOST}:{DB_PORT}/{db_name}"

ojo_engine = create_engine(get_url(os.getenv('OJO_DATABASE')))
analysis_engine = create_engine(get_url(os.getenv('ANALYSIS_DATABASE')))

# def get_engine(db_name_env_key):
#     user = os.environ['RDS_USERNAME']
#     password = os.environ['RDS_PASSWORD']
#     host = os.environ['RDS_HOST']
#     port = os.environ.get('RDS_PORT', '3306') # 포트는 관례상 3306 유지
#     db_name = os.environ[db_name_env_key]
    
#     url = f"mysql+pymysql://{user}:{password}@{host}:{port}/{db_name}?charset=utf8mb4"
    
#     return create_engine(
#         url,
#         poolclass=QueuePool,
#         pool_size=10,        # 분석 쿼리 동시 실행을 고려한 크기
#         max_overflow=20,     # 트래픽 급증 시 허용할 추가 커넥션
#         pool_recycle=1800,   # RDS 연결 끊김(3600초) 방지를 위해 30분마다 재연결
#         pool_pre_ping=True   # 커넥션 사용 전 유효성 체크 (가장 중요)
#     )

# ojo_engine = get_engine('OJO_DATABASE')
# analysis_engine = get_engine('ANALYSIS_DATABASE')
