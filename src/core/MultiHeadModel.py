import torch
import torch.nn as nn
import torchvision.models as models

class MultiHeadModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        # 공통 백본 (ResNet18)
        self.backbone = models.resnet18(pretrained=True)
        # 마지막 fully connected 레이어 제거
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # 특징 추출기
        self.feature_extractor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 분류 헤드 (night/day 분류)
        self.classification_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
        # 객체 검출 헤드 (YOLO 스타일)
        self.detection_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 5)  # x, y, w, h, confidence
        )
        
    def forward(self, x):
        # 공통 특징 추출
        batch_size = x.size(0)
        features = self.backbone(x)
        features = features.view(batch_size, -1)
        features = self.feature_extractor(features)
        
        # 병렬적으로 두 태스크 수행
        classification_output = self.classification_head(features)
        detection_output = self.detection_head(features)
        
        return {
            'classification': classification_output,
            'detection': detection_output
        } 