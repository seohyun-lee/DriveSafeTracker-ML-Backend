# # 독립 테스트 스크립트 예시 (예: test_s3.py)
# import os
# from dotenv import load_dotenv
# from utils.s3_utils import S3Uploader, make_filename

# load_dotenv()

# # 임의의 이미지 데이터
# dummy_image_data = b"your_image_data_here" # 실제 이미지 바이트 데이터로 대체

# try:
#     uploader = S3Uploader()
#     test_file_name = make_filename("website/test_uploads")
#     test_url = uploader.upload_file(dummy_image_data, file_name=test_file_name)
#     print(f"Test upload successful: {test_url}")
# except Exception as e:
#     print(f"Test upload failed: {e}")
