import cv2

class VehicleCandidateDetector:
    """
    Contour'lardan araç adayı çıkarır:
    - bounding box
    - centroid
    - basit şekil filtreleri (opsiyonel)
    """
    def __init__(self, min_w=25, min_h=25, max_w=None, max_h=None):
        self.min_w = int(min_w)
        self.min_h = int(min_h)
        self.max_w = int(max_w) if max_w is not None else None
        self.max_h = int(max_h) if max_h is not None else None

    def detect(self, contours, roi_frame, roi_offset=(0, 0)):
        """
        contours: ROI içinde bulunan contour listesi
        roi_frame: roi_cropped görüntüsü (sadece ROI alanı)
        roi_offset: (x1, y1) -> ROI'nin orijinal frame içindeki offset'i

        return:
          detections: [{'bbox': (x,y,w,h), 'centroid': (cx,cy), 'area': area}]
          debug_roi: roi_frame üstüne çizilmiş debug görüntü
        """
        x_off, y_off = roi_offset
        debug_roi = roi_frame.copy()
        detections = []

        for c in contours:
            x, y, w, h = cv2.boundingRect(c)

            # Basit boyut filtresi
            if w < self.min_w or h < self.min_h:
                continue
            if self.max_w is not None and w > self.max_w:
                continue
            if self.max_h is not None and h > self.max_h:
                continue

            area = w * h
            cx = x + w // 2
            cy = y + h // 2

            # ROI debug üzerinde çiz
            cv2.rectangle(debug_roi, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.circle(debug_roi, (cx, cy), 4, (0, 0, 255), -1)

            detections.append({
                "bbox": (x + x_off, y + y_off, w, h),         # full-frame coords
                "centroid": (cx + x_off, cy + y_off),         # full-frame coords
                "area": area
            })

        return detections, debug_roi
