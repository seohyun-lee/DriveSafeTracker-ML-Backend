import os, glob

# — 사용자 환경에 맞춰 수정 —
base_dir   = '/Users/leeseohyun/Documents/GitHub/DE-Project2-ML-Backend/dataset'
label_dir  = os.path.join(base_dir, 'labels', 'train')
image_dir  = os.path.join(base_dir, 'images', 'train')
mask_dir   = os.path.join(base_dir, 'masks', 'train')

# 1) labels/train의 basename(set) 생성
label_stems = {
    os.path.splitext(f)[0]
    for f in os.listdir(label_dir)
    if f.lower().endswith('.txt')
}

# 2) orphan 파일 탐색 및 dry-run
def find_orphans(dir_path, patterns):
    orphans = []
    for pat in patterns:
        for path in glob.glob(os.path.join(dir_path, pat)):
            stem = os.path.splitext(os.path.basename(path))[0]
            if stem not in label_stems:
                orphans.append(path)
    return orphans

print("=== Dry-run: 삭제 대상 파일 목록 ===")
print("\n-- images/train --")
for p in find_orphans(image_dir, ['*.jpg','*.jpeg','*.png']):
    print("  ", p)
# print("\n-- masks/train --")
# for p in find_orphans(mask_dir, ['*.png']):
#     print("  ", p)

# 3) 실제 삭제 (dry-run 후 확인되면 아래 주석 해제)
# for p in find_orphans(image_dir, ['*.jpg','*.jpeg','*.png']):
#     os.remove(p); print("Deleted:", p)
# for p in find_orphans(mask_dir, ['*.png']):
#     os.remove(p); print("Deleted:", p)
