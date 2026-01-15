import cv2

class ContourExtractor:
    def __init__(self, min_area=800, max_area=None):
        self.min_area = int(min_area)
        self.max_area = int(max_area) if max_area is not None else None

    def extract(self, binary_mask):
        """
        binary_mask: 0-255 tek kanal maske (Morphology sonrası)
        return: filtered_contours
        """
        # OpenCV sürümlerine göre findContours dönüşü değişebiliyor
        cnts = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = cnts[0] if len(cnts) == 2 else cnts[1]

        filtered = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.min_area:
                continue
            if self.max_area is not None and area > self.max_area:
                continue
            filtered.append(c)

        return filtered
