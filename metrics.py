import time
from collections import deque


class FPSMeter:
    """
    FPS ölçümü:
    - window_size kadar son frame süresinden anlık FPS hesaplar
    - avg_fps: pencere ortalaması
    """
    def __init__(self, window_size: int = 30):
        self.window_size = window_size
        self.times = deque(maxlen=window_size)
        self._last_t = None

    def tick(self) -> None:
        """Her frame başında çağır."""
        now = time.perf_counter()
        if self._last_t is not None:
            self.times.append(now - self._last_t)
        self._last_t = now

    def fps(self) -> float:
        """Pencere ortalamasına göre FPS."""
        if not self.times:
            return 0.0
        avg_dt = sum(self.times) / len(self.times)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0


class Stats:
    """
    Basit sayaç/metrik tutucu.
    Örn: contour sayısı, araç adayı sayısı vb.
    """
    def __init__(self):
        self.contours = 0
        self.candidates = 0
        self.counted = 0

    def reset_frame(self):
        self.contours = 0
        self.candidates = 0
