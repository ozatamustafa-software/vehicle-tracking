import cv2
import numpy as np

class MorphologyFilter:
    def __init__(self, kernel_size=5, iterations=1):
        self.kernel_size = kernel_size
        self.iterations = iterations
        self.kernel = np.ones((kernel_size, kernel_size), np.uint8)

    def apply(self, mask):
        # MOG2 shadow açıkken 127 değerleri üretebilir, bunu binary yapalım:
        _, mask_bin = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)

        # Opening: küçük gürültüleri temizler
        opened = cv2.morphologyEx(mask_bin, cv2.MORPH_OPEN, self.kernel, iterations=self.iterations)

        # Dilation: araç bölgelerini biraz büyütür, kopuklukları kapatır
        dilated = cv2.dilate(opened, self.kernel, iterations=1)

        return dilated

