from shapely.geometry import Polygon

PIXEL_TO_CM = 0.26
PIXEL_TO_M = PIXEL_TO_CM / 100

def polygon_area(polygon_coords):
    poly = Polygon(polygon_coords)
    area_pixels = poly.area
    return area_pixels * (PIXEL_TO_M ** 2)

def classify_risk(damage_type, area=None, length=None, width=None, count=None):
    if damage_type == "종방향":
        if count is not None and count <= 2 and length < 50:
            return "A"
        else:
            return "B"
    elif damage_type == "횡방향":
        if count is not None and length is not None:
            if count <= 2 and length < 20:
                return "A"
            else:
                return "B"
    elif damage_type == "거북등":
        if area is not None:
            if area < 0.1:
                return "A"
            else:
                return "B"
    elif damage_type == "불량 보수":
        if area is not None:
            if area < 0.1:
                return "A"
            else:
                return "B"
    elif damage_type == "포트홀":
        if width is not None:
            if width < 15:
                return "A"
            elif width < 30:
                return "B"
            else:
                return "C"
    elif damage_type == "젖은 도로":
        if area is not None:
            return "C"
    elif damage_type == "쓰레기":
        if width is not None:
            if width < 10:
                return "A"
            elif width < 30:
                return "B"
            else:
                return "C"
    elif damage_type == "Others":
        if area is not None:
            return "A"
    elif damage_type == "차선 손상":
        if area is not None:
            return "A"
    elif damage_type == "차선":
        return "-"
    return "정보 부족"

def summarize_image_risk(risks):
    count = len(risks)
    has_c = 'C' in risks
    has_b = 'B' in risks
    if has_c or count >= 4:
        return 'C'
    elif has_b or count >= 2:
        return 'B'
    else:
        return 'A' 