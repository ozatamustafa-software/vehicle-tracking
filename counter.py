class LineCounter:
    def __init__(self, line_y, offset=5):
        """
        line_y : sayma çizgisinin y koordinatı
        offset : tolerans (titreşim için)
        """
        self.line_y = line_y
        self.offset = offset
        self.count = 0
        self.tracked = set()  # sayılan centroid'ler

    def update(self, detections):
        """
        detections: [(cx, cy), (cx, cy), ...]
        """
        for cx, cy in detections:
            key = (cx, cy // 10)  # basit stabilizasyon

            # çizgiyi geçme kontrolü
            if self.line_y - self.offset < cy < self.line_y + self.offset:
                if key not in self.tracked:
                    self.count += 1
                    self.tracked.add(key)

        return self.count
