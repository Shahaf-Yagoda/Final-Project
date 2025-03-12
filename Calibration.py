import cv2
import numpy as np
import glob

# Termination criteria for corner refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), ..., (8,5,0)
objp = np.zeros((6 * 9, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

# Arrays to store object points and image points from all the images
objpoints = []  # 3d points in real world space
imgpoints = []  # 2d points in image plane

# Read images
images = glob.glob('calibration_images/*.jpg')

# Placeholder for the grayscale image
gray = None

for fname in images:
    img = cv2.imread(fname)

    if img is None:
        print(f"Failed to load image: {fname}")
        continue

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

    # If found, add object points, image points (after refining them)
    if ret:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (9, 6), corners2, ret)
        cv2.imshow('img', img)
        cv2.waitKey(500)
    else:
        print(f"Chessboard corners not found in image: {fname}")

cv2.destroyAllWindows()

if len(objpoints) > 0 and len(imgpoints) > 0:
    # Calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    # Save the camera calibration results
    np.save('camera_mtx.npy', mtx)
    np.save('camera_dist.npy', dist)
    np.save('camera_rvecs.npy', rvecs)
    np.save('camera_tvecs.npy', tvecs)

    print("Calibration successful.")
else:
    print("Calibration failed. Not enough valid images.")
