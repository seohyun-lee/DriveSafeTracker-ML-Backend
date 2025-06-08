import torch
import torchvision.transforms as transforms
from PIL import Image
from typing import Dict

class ModelWrapper:
    def __init__(self, model_path: str, device: str = "cpu"):
        self.device = torch.device(device)
        self.model = self._load_model(model_path)
        self.model.eval()

    def _load_model(self, path: str) -> torch.nn.Module:
        # TODO: 실제 아키텍처로 교체
        model = torch.hub.load("pytorch/vision:v0.14.0", "resnet18", pretrained=False)
        model.fc = torch.nn.Linear(model.fc.in_features, 3)  # 클래스 수 맞춤
        state = torch.load(path, map_location=self.device)
        model.load_state_dict(state)
        return model.to(self.device)

    def predict(self, image: Image.Image) -> Dict[str, float]:
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
        ])
        tensor = preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        classes = ["pothole", "debris", "clear"]
        return {cls: float(probs[i]) for i, cls in enumerate(classes)}