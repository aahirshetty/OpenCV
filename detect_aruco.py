import cv2
import numpy as np
import os

# Setup: 7x7 or 9x6 checkerboard are standard. 
# Count the internal corners!
CHESSBOARD_SIZE = (9, 6)
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

objp = np.zeros((CHESSBOARD_SIZE[0] * CHESSBOARD_SIZE[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_SIZE[0], 0:CHESSBOARD_SIZE[1]].T.reshape(-1, 2)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_corners, corners = cv2.findChessboardCorners(gray, CHESSBOARD_SIZE, None)

    if ret_corners:
        cv2.drawChessboardCorners(frame, CHESSBOARD_SIZE, corners, ret_corners)
        cv2.imshow('Calibration', frame)
        if cv2.waitKey(1) & 0xFF == ord('s'): # Press 's' to save a frame
            objpoints.append(objp)
            imgpoints.append(corners)
            print(f"Captured {len(objpoints)} frames")
    
    cv2.imshow('Calibration', frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or len(objpoints) > 30:
        break

# Calibrate
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# Save the calibration for the detection script
np.savez("calibration_data.npz", mtx=mtx, dist=dist)
print("Calibration complete. File saved as calibration_data.npz")
cap.release()
cv2.destroyAllWindows()
