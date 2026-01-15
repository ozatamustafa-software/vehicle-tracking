import cv2
import numpy as np

class ROISelector:
    def __init__(self, x=0, y=200, w=640, h=280):
        self.x, self.y, self.w, self.h = x, y, w, h
        self.last_rect = (0, 0, 0, 0)  # (x1,y1,x2,y2)

    def apply(self, frame):
        H, W = frame.shape[:2]
        x1 = max(0, self.x)
        y1 = max(0, self.y)
        x2 = min(W, self.x + self.w)
        y2 = min(H, self.y + self.h)

        self.last_rect = (x1, y1, x2, y2)

        # 1) Maskeli ROI
        mask = np.zeros((H, W), dtype=np.uint8)
        mask[y1:y2, x1:x2] = 255
        roi_masked = cv2.bitwise_and(frame, frame, mask=mask)

        # 2) Cropped ROI
        roi_cropped = frame[y1:y2, x1:x2].copy()

        # Debug görüntü
        debug = frame.copy()
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 0), 2)

        return roi_masked, roi_cropped, debug
