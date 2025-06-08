import os
from PIL import Image, ImageDraw

# 1) 경로 설정
img_dir = '/Users/leeseohyun/Documents/GitHub/DE-Project2-ML-Backend/dataset/images/val'
txt_dir = '/Users/leeseohyun/Documents/GitHub/DE-Project2-ML-Backend/dataset/labels/val'  # 원본 YOLO seg‐txt
mask_dir = '/Users/leeseohyun/Documents/GitHub/DE-Project2-ML-Backend/dataset/labels2/val'  # 생성할 PNG 마스크

os.makedirs(mask_dir, exist_ok=True)


# 2) 클래스 ID → 픽셀값 매핑 (원래 클래스 ID 그대로 쓰면 됩니다)
#    e.g. 클래스 ID 0~12 → 픽셀값 0~12
def cls_to_val(cls_id: int) -> int:
    return cls_id


# 3) 각 이미지별로 .txt 읽어 마스크 생성
for fname in os.listdir(txt_dir):
    if not fname.endswith('.txt'):
        continue
    base = os.path.splitext(fname)[0]
    img_path = os.path.join(img_dir, base + '.jpg')
    txt_path = os.path.join(txt_dir, fname)

    # 3-1) 이미지 크기 얻기
    with Image.open(img_path) as im:
        W, H = im.size

    # 3-2) 빈 마스크 (L 모드, 0=background)
    mask = Image.new('L', (W, H), 0)
    draw = ImageDraw.Draw(mask)

    # 3-3) polygon 데이터 파싱
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            cls_id = int(float(parts[0]))
            coords = list(map(float, parts[1:]))
            # normalized → pixel 좌표 변환
            pts = [
                (coords[i] * W, coords[i + 1] * H)
                for i in range(0, len(coords), 2)
            ]
            # 채우기
            draw.polygon(pts, fill=cls_to_val(cls_id))

    # 3-4) PNG로 저장
    mask.save(os.path.join(mask_dir, base + '.png'))
    print(f"Saved mask for {base}")
