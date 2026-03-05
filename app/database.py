import os
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv()

# 환경 변수가 없으면 로컬 환경(127.0.0.1:3307)을 바라보도록 설정
DB_HOST = os.getenv("DB_HOST", "127.0.0.1")
DB_PORT = os.getenv("DB_PORT", "3306") 

def get_url(db_name):
    return f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{DB_HOST}:{DB_PORT}/{db_name}"

ojo_engine = create_engine(get_url(os.getenv('OJO_DATABASE')))
analysis_engine = create_engine(get_url(os.getenv('ANALYSIS_DATABASE')))