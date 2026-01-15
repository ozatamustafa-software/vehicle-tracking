from ultralytics import YOLO


class YOLODetector:
    """
    Ultralytics YOLOv8 detector wrapper.

    COCO class id'leri:
      car=2, motorcycle=3, bus=5, truck=7
    """

    def __init__(self, model_name="yolov8n.pt", conf=0.35, classes=None, device=None):
        self.model = YOLO(model_name)
        self.conf = conf
        self.classes = classes
        self.device = device

    def detect(self, frame_bgr):
        results = self.model.predict(
            source=frame_bgr,
            conf=self.conf,
            classes=self.classes,
            device=self.device,
            verbose=False
        )

        detections = []

        if not results or results[0].boxes is None:
            return detections

        r = results[0]

        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0].item())
            cls = int(box.cls[0].item())
            name = r.names.get(cls, str(cls))

            detections.append({
                "bbox": (x1, y1, x2, y2),
                "conf": conf,
                "cls": cls,
                "name": name,
                "centroid": ((x1 + x2) // 2, (y1 + y2) // 2)
            })

        return detections
