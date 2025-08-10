import numpy as np

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba, bc = a - b, c - b
    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def get_rula_score(angle):
    if angle <= 20:
        return 1
    elif angle <= 45:
        return 2
    elif angle <= 90:
        return 3
    else:
        return 4

def get_risk_label(rula_score):
    if rula_score <= 2:
        return "Low"
    elif rula_score <= 4:
        return "Medium"
    else:
        return "High"