import csv
import os
from datetime import datetime


class MetricsLogger:
    """
    Her çalıştırmada bir CSV oluşturur ve frame bazlı özetleri yazar.
    Rapor için kanıt olur: FPS, detections, tracks, counts...
    """

    def __init__(self, out_dir="logs", filename_prefix="run"):
        os.makedirs(out_dir, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(out_dir, f"{filename_prefix}_{ts}.csv")

        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.writer(self._file)
        self._writer.writerow([
            "frame_id",
            "fps",
            "num_detections",
            "num_tracks",
            "total",
            "up",
            "down"
        ])

    def write(self, frame_id, fps, num_dets, num_tracks, total, up, down):
        self._writer.writerow([
            int(frame_id),
            float(fps),
            int(num_dets),
            int(num_tracks),
            int(total),
            int(up),
            int(down)
        ])

    def close(self):
        try:
            self._file.close()
        except Exception:
            pass
