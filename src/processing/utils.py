# src/processing/utils.py
import cv2
import numpy as np


def calculate_angle(a, b, c):
    """
    Calculates the angle at point 'b' formed by three points a, b, and c.
    
    Geometrically, this is the angle between vectors:
        - Vector ba: from b to a
        - Vector bc: from b to c

    Parameters:
        a, b, c: Lists or tuples representing points (x, y) or (x, y, z).
                 Only the x and y coordinates are used.

    Returns:
        The angle in degrees between vector ba and vector bc.
    """

    # Convert points to numpy arrays and extract only x, y
    a = np.array(a[:2])
    b = np.array(b[:2])
    c = np.array(c[:2])

    # Calculate vectors
    ba = a - b
    bc = c - b

    # Compute dot product and magnitudes
    dot = np.dot(ba, bc)
    mag_ba = np.linalg.norm(ba)
    mag_bc = np.linalg.norm(bc)

    # Prevent division by zero
    if mag_ba * mag_bc == 0:
        return 0.0

    # Compute cosine of the angle
    cosine_angle = dot / (mag_ba * mag_bc)

    # Clamp value to valid range [-1, 1] to avoid floating-point errors
    cosine_angle = max(min(cosine_angle, 1.0), -1.0)

    # Return the angle in degrees
    angle = np.degrees(np.arccos(cosine_angle))
    return angle

def draw_joint_angle(image, a, b, c, angle, min_ok, max_ok, label="", override_color=None):
    """
    Draws lines between points a–b and b–c and labels the angle at point b.
    - a, b, c: lists of [x, y] (normalized)
    - angle: numeric angle in degrees
    - min_ok, max_ok: range considered 'correct'
    """
    h, w = image.shape[:2]
    pt_a = (int(a[0] * w), int(a[1] * h))
    pt_b = (int(b[0] * w), int(b[1] * h))
    pt_c = (int(c[0] * w), int(c[1] * h))

    color = override_color if override_color else (
        (0, 255, 0) if min_ok <= angle <= max_ok else (0, 0, 255)
    )

    cv2.line(image, pt_a, pt_b, color, 4)
    cv2.line(image, pt_b, pt_c, color, 4)

    text = f"{label}{angle:.1f}°"
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)

    cv2.rectangle(image,
                  (pt_b[0], pt_b[1] - text_height - 10),
                  (pt_b[0] + text_width + 10, pt_b[1] + 5),
                  (0, 0, 0), -1)

    cv2.putText(image, text, (pt_b[0] + 5, pt_b[1] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)


def distance_2d(a, b):
    """
    Returns 2D distance between points A and B.
    Each input is [x, y] or [x, y, z].
    """
    a = np.array(a[:2])
    b = np.array(b[:2])
    return np.linalg.norm(a - b)