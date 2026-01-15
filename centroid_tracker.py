import numpy as np
from scipy.spatial import distance as dist


class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=50):
        """
        max_disappeared: Bir obje kaç frame görünmezse silinsin
        max_distance:    Eşleştirmede izin verilen maksimum centroid mesafesi (piksel)
        """
        self.next_id = 0
        self.objects = {}      # id -> {"centroid": (cx, cy), "bbox": (x1,y1,x2,y2)}
        self.disappeared = {}  # id -> frames
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    # ✅ EKLENDİ: bbox parametresi (geriye uyumlu olsun diye bbox=None)
    def register(self, centroid, bbox=None):
        self.objects[self.next_id] = {
            "centroid": centroid,
            "bbox": bbox
        }
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        if object_id in self.objects:
            del self.objects[object_id]
        if object_id in self.disappeared:
            del self.disappeared[object_id]

    def update(self, detections):
        """
        detections: [(x1,y1,x2,y2), ...]
        returns: dict[id] = {"centroid": (cx, cy), "bbox": (x1,y1,x2,y2)}
        """

        # 1) Hiç detection yoksa: mevcut objeleri "disappeared" artır
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    self.deregister(object_id)
            return self.objects

        # 2) Gelen bbox'ları centroid'e çevir
        input_centroids = np.zeros((len(detections), 2), dtype="int")
        for i, (x1, y1, x2, y2) in enumerate(detections):
            cx = int((x1 + x2) / 2)
            cy = int((y1 + y2) / 2)
            input_centroids[i] = (cx, cy)

        # 3) Takipte hiç obje yoksa hepsini register et
        if len(self.objects) == 0:
            for i in range(len(input_centroids)):
                # ✅ EKLENDİ: register'a bbox da ver
                self.register(input_centroids[i], detections[i])
            return self.objects

        # 4) Mevcut objeler ile yeni centroid'ler arası mesafe matrisi
        object_ids = list(self.objects.keys())

        # ✅ DEĞİŞTİ: objects artık dict tuttuğu için centroid listesi çıkarıyoruz
        object_centroids = np.array([self.objects[obj_id]["centroid"] for obj_id in object_ids])

        D = dist.cdist(object_centroids, input_centroids)

        # En küçük mesafeler üzerinden greedy eşleştirme
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows = set()
        used_cols = set()

        # 5) Eşleştirme + max_distance filtresi
        for row, col in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue

            # ✅ Kritik kısım: Çok uzaksa eşleştirme yapma
            if D[row, col] > self.max_distance:
                continue

            object_id = object_ids[row]

            # ✅ DEĞİŞTİ: hem centroid hem bbox güncelleniyor
            self.objects[object_id] = {
                "centroid": input_centroids[col],
                "bbox": detections[col]
            }
            self.disappeared[object_id] = 0

            used_rows.add(row)
            used_cols.add(col)

        # 6) Eşleşmeyen eski objeler -> disappeared artır
        unused_rows = set(range(D.shape[0])) - used_rows
        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                self.deregister(object_id)

        # 7) Eşleşmeyen yeni detection'lar -> yeni obje register
        unused_cols = set(range(len(input_centroids))) - used_cols
        for col in unused_cols:
            # ✅ EKLENDİ: yeni objeye bbox ver
            self.register(input_centroids[col], detections[col])

        return self.objects
