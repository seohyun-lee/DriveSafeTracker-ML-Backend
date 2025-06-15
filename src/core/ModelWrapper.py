import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Dict, List
import os
from ultralytics import YOLO

class ModelWrapper:
    def __init__(self, model_paths: List[str], device: str = "cpu"):
        self.device = torch.device(device)
        self.models = self._load_models(model_paths)
        for model in self.models:
            if not isinstance(model, YOLO):  # YOLO 모델이 아닌 경우에만 eval() 호출
                model.eval()

    def _load_models(self, paths: List[str]) -> List[torch.nn.Module]:
        models = []
        for path in paths:
            model_path = os.path.join("src", "models", path)
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {model_path}")
            
            if path.endswith('.pt'):  # YOLO 모델
                try:
                    # ultralytics YOLO 직접 사용
                    model = YOLO(model_path)
                except Exception as e:
                    print(f"YOLO 모델 로딩 실패: {str(e)}")
                    raise
            else:  # 일반 PyTorch 모델
                model = torch.hub.load("pytorch/vision:v0.14.0", "resnet18", pretrained=False)
                model.fc = torch.nn.Linear(model.fc.in_features, 2)
                state = torch.load(model_path, map_location=self.device)
                model.load_state_dict(state)
            
            models.append(model)
        return models

    def predict(self, image: Image.Image) -> Dict:
        results = {}
        
        # 첫 번째 모델 (night_day_model) 예측
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        tensor = preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.models[0](tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        classes = ["day", "night"]
        results["classification"] = {cls: float(probs[i]) for i, cls in enumerate(classes)}

        # 두 번째 모델 (YOLO) 예측
        yolo_results = self.models[1](image)  # YOLO 모델은 PIL Image를 직접 받을 수 있음
        # YOLO 결과를 필요한 형식으로 변환
        yolo_predictions = []
        for result in yolo_results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                conf = box.conf.cpu().numpy()[0]
                cls = int(box.cls.cpu().numpy()[0])
                yolo_predictions.append({
                    "bbox": [int(x1), int(y1), int(x2-x1), int(y2-y1)],
                    "confidence": float(conf),
                    "class": cls
                })
        results["detection"] = {"detections": yolo_predictions}

        return results