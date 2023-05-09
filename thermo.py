import cv2
import numpy as np

cap = cv2.VideoCapture("Rec-000260.seq")

def preprocess(frame):
    equalized = cv2.equalizeHist(frame)
    normalized = cv2.normalize(equalized, None, alpha=0, beta=255,
                    norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
    return normalized

def identify_head(frame):
    edges = cv2.Canny(frame, 100, 200)
    _, contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    head_contour = max(contours, key=cv2.contourArea)
    mask = np.zeros_like(frame)
    cv2.fillPoly(mask, [head_contour], 255)
    return mask

prev_gray = None
head_center = None
def track_motion(frame):
    global prev_gray, head_center
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                        0.5, 3, 15, 3, 5, 1.2, 0)
    prev_gray = gray
    if head_center is None:
        moments = cv2.moments(head_mask)
        head_center = (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
    else:
        head_center = (head_center[0] + flow[int(head_center[1]), int(head_center[0])][0],
                       head_center[1] + flow[int(head_center[1]), int(head_center[0])][1])
    return head_center

def compensate_motion(frame, head_center):
    height, width = frame.shape[:2]
    translation_matrix = np.float32([[1, 0, -head_center[0]], [0, 1, -head_center[1]]])
    frame = cv2.warpAffine(frame, translation_matrix, (width, height))
    return frame

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = preprocess(frame)
    head_mask = identify_head(frame)
    head_center = track_motion(frame)
    compensated_frame = compensate_motion(frame, head_center)
    cv2.imshow("Kompensacja", compensated_frame)
    if cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()





