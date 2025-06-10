import os

def clamp(val: float) -> float:
    return max(0.0, min(1.0, val))

def process_label_file(file_path: str):
    with open(file_path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        tokens = line.strip().split()
        if not tokens:
            continue
        cls = tokens[0]
        coords = tokens[1:]
        try:
            clamped_coords = [str(clamp(float(v))) for v in coords]
            new_line = f"{cls} " + " ".join(clamped_coords)
            new_lines.append(new_line)
        except ValueError:
            print(f"⚠️ Invalid value in {file_path}: {line.strip()}")
            continue

    with open(file_path, "w") as f:
        for line in new_lines:
            f.write(line + "\n")

def process_all_labels(root_dir: str):
    for subdir, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(subdir, file)
                process_label_file(file_path)
                print(f"✅ Processed: {file_path}")

# 실행 경로 (원하는 경로로 수정 가능)
label_root = "/Users/leeseohyun/Documents/GitHub/DE-Project2-ML-Backend/dataset/labels"
process_all_labels(label_root)
