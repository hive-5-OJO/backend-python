import os
import urllib.parse
from dotenv import load_dotenv
from sqlalchemy import create_engine

# 1. 현재 database.py 파일이 있는 폴더의 절대 경로를 찾습니다.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. 그 폴더 안에 있는 .env 파일의 경로를 지정합니다.
env_path = os.path.join(BASE_DIR, '.env')

# 3. 명확하게 해당 경로의 .env 파일을 읽어오라고 지시합니다.
load_dotenv(dotenv_path=env_path)

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST")
DB_PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

# ★ 안전장치: 만약 .env 파일을 못 읽어서 값이 비어있다면 친절한 에러를 띄웁니다.
if DB_PASSWORD is None:
    raise ValueError("\n[에러] .env 파일을 읽지 못했습니다! \n1. .env 파일이 app 폴더 안에 있는지 확인하세요. \n2. 파일 이름이 .env.txt 로 되어있지 않은지 확인하세요.")

# 비밀번호 특수문자 처리 및 URL 조립
encoded_password = urllib.parse.quote_plus(DB_PASSWORD)
DB_URL = f"mysql+pymysql://{DB_USER}:{encoded_password}@{DB_HOST}:{DB_PORT}/{DB_NAME}?charset=utf8mb4"

# 데이터베이스 엔진 생성
engine = create_engine(DB_URL)