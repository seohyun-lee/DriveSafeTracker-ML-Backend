from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
from uuid import uuid4
import io
from PIL import Image
import numpy as np

from src.core.PredictionItem import BBoxPrediction, PolygonPrediction, Point, PredictionItem
from src.core.ModelWrapper import ModelWrapper
from src.utils.s3_utils import S3Uploader, make_filename
from src.utils.risk_utils import classify_risk, summarize_image_risk, PIXEL_TO_CM, PIXEL_TO_M

app = FastAPI(
    title="Road Hazard Classification API",
    description="도로 노면 이미지 내 위험물 분류 서비스",
    version="1.0"
)

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

DAMAGE_CLASSES = {
    0: '기타',
    1: '거북등 균열',
    2: '낮',
    3: '밤',
    4: '불량 보수',
    5: '쓰레기',
    6: '양호',
    7: '젖은 도로',
    8: '종방향 균열',
    9: '차선',
    10: '차선 손상',
    11: '포트홀',
    12: '횡방향 균열'
}

@app.on_event("startup")
async def startup_event():
    app.state.model = ModelWrapper(
        model_paths=["night_day_model.pth", "yolo_best.pt"],
        device="cpu"
    )
    app.state.s3_uploader = S3Uploader()

@app.get("/", tags=["health"])
async def health_check():
    return {"status": "ok", "message": "Service is up and running"}

@app.post("/predict")
async def predict_hazard(file: UploadFile = File(...)):
    try:
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="이미지 파일을 업로드하세요.")
        
        data = await file.read()
        try:
            image = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"이미지 파싱에 실패했습니다: {str(e)}")

        # 1. 원본 이미지 S3 업로드
        upload_file_name = make_filename("website/uploads")
        s3_url_upload = app.state.s3_uploader.upload_file(data, file_name=upload_file_name)

        # 2. 예측 및 바운딩박스 시각화
        predictions = app.state.model.predict(image)
        predictions_for_frontend = []
        risk_list = []
        # 분류 결과: 인덱스 기준 한글 클래스명 매핑
        class_probs = list(predictions["classification"].values())
        max_idx = int(np.argmax(class_probs))
        max_conf = class_probs[max_idx]
        day_or_night = DAMAGE_CLASSES.get(max_idx, "알수없음") if max_idx in [2, 3] else "알수없음"
        # YOLO 검출 결과: class id 기준 한글 클래스명 매핑
        for detection in predictions["detection"]["detections"]:
            class_id = detection["class"]
            # 2(낮), 3(밤), 6(양호)는 제외, 9(차선)는 포함
            if class_id in [2, 3, 6]:
                continue
            if detection["confidence"] > 0.5:
                bbox = detection["bbox"]
                class_name = DAMAGE_CLASSES.get(class_id, "알수없음")
                w = bbox[2] * PIXEL_TO_CM
                h = bbox[3] * PIXEL_TO_CM
                area_m2 = (bbox[2] * bbox[3]) * (PIXEL_TO_M ** 2)
                risk_level = classify_risk(class_name, area=area_m2, width=w, length=h)
                risk_list.append(risk_level)
                predictions_for_frontend.append({
                    "class_id": class_id,
                    "name": class_name,
                    "confidence": detection["confidence"],
                    "type": "bbox",
                    "x": int(bbox[0]),
                    "y": int(bbox[1]),
                    "width": int(bbox[2]),
                    "height": int(bbox[3]),
                    "width_cm": round(w, 1),
                    "length_cm": round(h, 1),
                    "area_m2": round(area_m2, 3),
                    "risk_level": risk_level
                })
        overall_risk = summarize_image_risk(risk_list)

        # 3. 바운딩박스 시각화 이미지 생성 (YOLO 결과 활용)
        yolo_results = app.state.model.models[1](image)
        result = yolo_results[0]
        img_with_boxes = result.plot()  # numpy array
        img_with_boxes_pil = Image.fromarray(img_with_boxes)
        buf = io.BytesIO()
        img_with_boxes_pil.save(buf, format='JPEG')
        buf.seek(0)
        result_img_bytes = buf.getvalue()

        # 4. 결과 이미지 S3 업로드
        result_file_name = make_filename("website/results")
        s3_url_result = app.state.s3_uploader.upload_file(result_img_bytes, file_name=result_file_name)

        # 5. 결과 이미지 프론트엔드 전송 (S3 URL)
        return {
            "predictions": predictions_for_frontend,
            "day_or_night": day_or_night,
            "overall_risk": overall_risk,
            "original_image_url": s3_url_upload,
            "result_image_url": s3_url_result
        }
    except Exception as e:
        print(f"예측 중 에러 발생: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# 실행
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host="127.0.0.1", port=8000, reload=True)
