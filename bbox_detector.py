import cv2

class BBoxDetector:
    """
    Clean binary mask -> contours -> bounding boxes
    """
    def __init__(self, min_area=500):
        self.min_area = min_area

    def detect(self, mask, draw_on=None):
        """
        mask: binary image (0/255)
        draw_on: optional BGR frame to draw boxes on
        return: (boxes, drawn_frame)
            boxes: [(x, y, w, h), ...]
        """
        # OpenCV bazen maskeyi 0/255 bekler, emin olalım:
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

        # Contour bul
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        out = None
        if draw_on is not None:
            out = draw_on.copy()

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append((x, y, w, h))

            if out is not None:
                cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return boxes, out
