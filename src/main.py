from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, List
from uuid import uuid4
import io
from PIL import Image

from src import PredictionItem
from src.PredictionItem import BBoxPrediction, PolygonPrediction, Point

app = FastAPI(
    title="Road Hazard Classification API",
    description="도로 노면 이미지 내 위험물 분류 서비스",
    version="1.0"
)

# CORS 설정 (필요에 따라 origin 제한)
app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 루트 헬스체크
@app.get("/", tags=["health"])
async def health_check():
    return {"status": "ok", "message": "Service is up and running"}

# @app.on_event("startup")
# async def load_model():
#     # 애플리케이션 전역 상태에 모델 저장
#     app.state.model = ModelWrapper(model_path="model.pth", device="cpu")

# 위에 정의한 Pydantic 모델들을 여기에 포함시키거나 import 합니다.
# class Point(BaseModel): ...
# class BasePrediction(BaseModel): ...
# class BBoxPrediction(BasePrediction): ...
# class PolygonPrediction(BasePrediction): ...
# PredictionItem = Union[BBoxPrediction, PolygonPrediction]



# 예시: 모델의 predict 함수가 반환하는 값의 형태를 가정합니다.
# 실제 모델의 출력에 맞게 이 부분을 수정해야 합니다.
def process_model_predictions(model_preds) -> List[PredictionItem]:
    processed_results = []
    for pred in model_preds: # model_preds가 리스트 형태의 예측 결과라고 가정
        # --- 여기부터는 모델의 실제 출력에 따라 크게 달라집니다 ---
        # 예시: pred가 딕셔너리 형태이고, 필요한 정보를 포함한다고 가정
        # pred = {"label": "pothole", "confidence": 0.92, "box": [100, 150, 50, 30], "type": "bbox"}
        # pred = {"label": "crack", "confidence": 0.88, "polygon_points": [[10,20], [15,30], [5,25]], "type": "polygon"}

        hazard_name = pred.get("label", "Unknown") # 모델 출력에서 라벨 가져오기
        confidence_score = pred.get("confidence")
        annotation_type = pred.get("type") # "bbox" 또는 "polygon"

        # 위험도 결정 로직 (예시)
        severity = "low"
        if hazard_name.lower() in ["pothole", "object_on_road"]:
            severity = "high"
        elif hazard_name.lower() in ["crack", "lane_departure"]:
            severity = "medium"

        item_id = str(uuid4())

        if annotation_type == "bbox" and "box" in pred:
            box = pred["box"] # [x, y, width, height] 형태라고 가정
            processed_results.append(
                BBoxPrediction(
                    id=item_id,
                    name=hazard_name,
                    severity=severity,
                    confidence=confidence_score,
                    x=box[0],
                    y=box[1],
                    width=box[2],
                    height=box[3],
                )
            )
        elif annotation_type == "polygon" and "polygon_points" in pred:
            # points가 [[x1,y1], [x2,y2], ...] 형태라고 가정
            points_data = [Point(x=p[0], y=p[1]) for p in pred["polygon_points"]]
            processed_results.append(
                PolygonPrediction(
                    id=item_id,
                    name=hazard_name,
                    severity=severity,
                    confidence=confidence_score,
                    points=points_data,
                )
            )
        # --- 모델 출력 처리 로직 끝 ---
    return processed_results

# response_model을 List[PredictionItem]으로 변경
@app.post("/predict", response_model=List[PredictionItem])
async def predict_hazard(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="이미지 파일을 업로드하세요.")
    data = await file.read()
    try:
        image = Image.open(io.BytesIO(data)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="이미지 파싱에 실패했습니다.")

    # app.state.model.predict(image)가 모델의 원시 예측 결과를 반환한다고 가정
    raw_preds = app.state.model.predict(image)

    # 모델 예측 결과를 프론트엔드 형식으로 변환
    # 이 부분은 실제 모델의 출력 형식에 맞춰 구현해야 합니다.
    # 예를 들어, raw_preds가 [{label: 'pothole', box: [x,y,w,h]}, ...] 형태일 수 있습니다.
    predictions_for_frontend = process_model_predictions(raw_preds)

    # JSONResponse를 직접 사용하는 대신, FastAPI가 response_model에 따라 자동으로 직렬화하도록 함
    return predictions_for_frontend

# 실행: uvicorn app:app --host 0.0.0.0 --port 8000
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
