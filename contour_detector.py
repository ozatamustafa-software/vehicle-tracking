import cv2

class ContourDetector:
    def __init__(self, min_area=800, draw=True):
        """
        min_area: Araç olmayan küçük nesneleri elemek için minimum contour alanı
        draw: çizim yapıp yapmama
        """
        self.min_area = min_area
        self.draw = draw

    def detect(self, clean_mask, roi_frame):
        """
        clean_mask: Morphology sonrası temiz maske (tek kanal)
        roi_frame: ROI cropped renkli görüntü (BGR)

        return:
          - detections: [(x, y, w, h, area), ...]
          - annotated: ROI üzerinde kutuları çizilmiş görüntü
        """
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        annotated = roi_frame.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            detections.append((x, y, w, h, area))

            if self.draw:
                cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(annotated, f"A:{int(area)}", (x, max(0, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return detections, annotated
