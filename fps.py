import time

class FPSMeter:
    def __init__(self, avg_window=30):
        self.avg_window = avg_window
        self.times = []
        self.last = None

    def tick(self):
        now = time.time()
        if self.last is not None:
            dt = now - self.last
            self.times.append(dt)
            if len(self.times) > self.avg_window:
                self.times.pop(0)
        self.last = now

    def fps(self):
        if not self.times:
            return 0.0
        avg_dt = sum(self.times) / len(self.times)
        return 1.0 / avg_dt if avg_dt > 0 else 0.0
