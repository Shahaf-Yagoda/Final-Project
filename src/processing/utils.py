# src/processing/utils.py
import numpy as np

def calculate_angle(a, b, c):
    a = np.array(a[:2])  # Only use x,y
    b = np.array(b[:2])
    c = np.array(c[:2])

    ba = a - b
    bc = c - b
    dot = np.dot(ba, bc)
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)

    if mag_ba * mag_bc == 0:
        return 0.0

    cosine_angle = dot / (mag_ba * mag_bc)
    cosine_angle = max(min(cosine_angle, 1.0), -1.0)
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def draw_joint_angle(image, a, b, c, angle, min_ok, max_ok, label="", override_color=None):
    # Function content to draw joint angle (same as before)
    pass  # Implement this function here, as done in the original code
