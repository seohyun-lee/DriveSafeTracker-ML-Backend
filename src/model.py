from ultralytics import YOLO

# 1) 모델 로드 & 학습
model = YOLO("yolo11n-seg.pt")
metrics = model.train(
    data="../dataset/coco.yaml",  # images/…만 지정, labels/*.png 자동 인식
    epochs=1,
    imgsz=64,
    task="segment"
)

# 2) Detection & Segmentation 메트릭 출력
# results.box  → Detection, results.seg → Segmentation
print("=== Detection Metrics ===")
print(f" mAP@0.5    : {metrics.box.map50:.3f}")
print(f" mAP@0.5-0.95: {metrics.box.map:.3f}")

print("\n=== Segmentation Metrics ===")
print(f" mAP@0.5    : {metrics.seg.map50:.3f}")
print(f" mAP@0.5-0.95: {metrics.seg.map:.3f}")

# 3) Inference & 결과 시각화
results = model.predict(
    source="/Users/leeseohyun/Documents/GitHub/DE-Project2-ML-Backend/dataset/images/val/G_A_F_01_0822092712516.jpg",
    imgsz=64,
    task="segment"
)
for i, r in enumerate(results):
    print(f"\n--- Prediction {i+1} ---")
    print(" Boxes:",    r.boxes.xyxy)
    print(" Confidences:", r.boxes.conf)
    print(" Classes:",    r.boxes.cls)
    if r.masks and r.masks.data is not None:
        print(" Masks shape:", r.masks.data.shape)
    else:
        print(" No masks detected")
    r.show()  # plot() → 화면에 bounding box+mask 오버레이


# 추가로, inference 시 반환되는 객체 구조도 확인해보겠습니다
# # 임의의 이미지 한 장으로 predict 해보기
# preds = model.predict(source="/Users/leeseohyun/Documents/GitHub/DE-Project2-ML-Backend/dataset/images/val/G0010139_JPG_jpg.rf.3d3f9990337d213d85f0fb1ef468d990.jpg", imgsz=64)
# print("\nType of preds:", type(preds), "  length:", len(preds))
# print("   → preds is a list of Results objects")
#
# # 첫 번째 결과 객체 내부 구조 확인
# r = preds[0]
# print("\nAttributes of a single Results object:", r.__dict__.keys())
#
# # Detection 박스 정보
# print("\nBounding boxes (xyxy, conf, cls):\n", r.boxes.xyxy, r.boxes.conf, r.boxes.cls)
#
# # Segmentation 마스크 정보
# print("\nMasks (shape):", r.masks.data.shape)
