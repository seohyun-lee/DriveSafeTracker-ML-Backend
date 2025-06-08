from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
import uuid # id 생성을 위해 추가

class BasePositionPrediction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    severity: Literal["high", "medium", "low"]
    confidence: Optional[float] = None
    # type 필드는 각 하위 클래스(BBoxPrediction, PolygonPrediction)에서 구체적으로 정의합니다.


class Point(BaseModel):
    x: int
    y: int

class BasePrediction(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    severity: Literal["high", "medium", "low"]
    confidence: Optional[float] = None

class BBoxPrediction(BasePositionPrediction):
    type: Literal["bbox"] = "bbox"
    x: int
    y: int
    width: int
    height: int

class PolygonPrediction(BasePositionPrediction):
    type: Literal["polygon"] = "polygon"
    points: List[Point]


# Union을 사용하여 bbox 또는 polygon 중 하나를 나타냄
PredictionItem = Union[BBoxPrediction, PolygonPrediction]
