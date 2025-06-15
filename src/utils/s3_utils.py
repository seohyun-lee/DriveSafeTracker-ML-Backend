import boto3
from botocore.client import Config
import os
from datetime import datetime
from typing import Optional
from dotenv import load_dotenv
import uuid
import logging

# 로거 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()

def make_filename(prefix: str, ext: str = "jpg") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    uid = str(uuid.uuid4())[:8]
    return f"{prefix}/{timestamp}_{uid}.{ext}"

class S3Uploader:
    def __init__(self):
        logger.info("S3Uploader 초기화 시작")
        self.aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        self.aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        self.aws_region = os.getenv('AWS_REGION')
        self.bucket_name = os.getenv('S3_BUCKET_NAME')

        logger.info(f"Loaded AWS_ACCESS_KEY_ID: {self.aws_access_key_id[:5]}...")
        logger.info(f"Loaded AWS_REGION: {self.aws_region}")
        logger.info(f"Loaded S3_BUCKET_NAME: {self.bucket_name}")

        if not self.aws_access_key_id or not self.aws_secret_access_key or not self.aws_region or not self.bucket_name:
            logger.error("필수 AWS 환경 변수가 설정되지 않았습니다.")
            raise ValueError("AWS 환경 변수(ACCESS_KEY_ID, SECRET_ACCESS_KEY, REGION, BUCKET_NAME)를 설정해야 합니다.")

        try:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=self.aws_access_key_id,
                aws_secret_access_key=self.aws_secret_access_key,
                region_name=self.aws_region,
                config=Config(s3={'addressing_style': 'path'})
            )
            logger.info("S3 클라이언트 초기화 성공.")
        except Exception as e:
            logger.error(f"S3 클라이언트 초기화 실패: {str(e)}")
            raise

    def upload_file(self, file_data: bytes, file_name: Optional[str] = None) -> str:
        logger.info("파일 S3 업로드 시도 중...")
        if file_name is None:
            file_name = make_filename("website/uploads")
            logger.info(f"파일 이름 자동 생성: {file_name}")
        
        logger.info(f"업로드 대상 버킷: {self.bucket_name}, 키: {file_name}")
        logger.info(f"파일 데이터 크기: {len(file_data)} bytes")

        try:
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=file_name,
                Body=file_data,
                ContentType='image/jpeg'
            )
            url = f"https://{self.bucket_name}.s3.{self.aws_region}.amazonaws.com/{file_name}"
            logger.info(f"파일 S3 업로드 성공. URL: {url}")
            return url
        except Exception as e:
            logger.error(f"S3 업로드 중 에러 발생: {str(e)}", exc_info=True)
            raise 