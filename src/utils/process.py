import os,glob
from PIL import Image, ImageDraw

# 1) 경로 설정
image_dir = '/Users/leeseohyun/Documents/GitHub/DE-Project2-ML-Backend/dataset/images/test'
label_dir = '/Users/leeseohyun/Documents/GitHub/DE-Project2-ML-Backend/dataset/labels/test'  # 원본 YOLO seg‐txt
mask_dir = '/Users/leeseohyun/Documents/GitHub/DE-Project2-ML-Backend/dataset/masks/test'  # 생성할 PNG 마스크

os.makedirs(mask_dir, exist_ok=True)

# 처리할 클래스 아이디 세트
TARGET_CLASSES = {1, 4, 7, 9, 10}

for lbl_path in glob.glob(os.path.join(label_dir, '*.txt')):
    name = os.path.splitext(os.path.basename(lbl_path))[0]
    # 이미지 경로 추정 (jpg 또는 png)
    for ext in ('.jpg', '.png'):
        img_path = os.path.join(image_dir, name + ext)
        if os.path.isfile(img_path):
            break
    else:
        continue  # 이미지가 없으면 건너뜀

    img = Image.open(img_path)
    w, h = img.size
    # 빈 흑백 마스크 (픽셀값=0)
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)

    with open(lbl_path) as f:
        for line in f:
            parts = line.strip().split()
            cls = int(parts[0])
            if cls not in TARGET_CLASSES:
                continue
            coords = list(map(float, parts[1:]))
            # 정규화 좌표 → 픽셀 좌표 리스트
            polygon = [(coords[i] * w, coords[i+1] * h)
                       for i in range(0, len(coords), 2)]
            # 클래스 아이디를 픽셀 값으로 채움
            draw.polygon(polygon, fill=cls)

    # PNG로 저장
    mask.save(os.path.join(mask_dir, name + '.png'))