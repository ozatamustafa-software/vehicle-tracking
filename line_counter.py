# src/core/line_counter.py
# C++ hızlandırma varsa onu kullanır, yoksa Python fallback.

try:
    # Derlenen .pyd dosyası src/cpp içindeyse, Python onu bulamayabilir.
    # Bu yüzden path ekliyoruz.
    import os, sys
    CPP_DIR = os.path.join(os.path.dirname(__file__), "..", "cpp")
    CPP_DIR = os.path.abspath(CPP_DIR)
    if CPP_DIR not in sys.path:
        sys.path.insert(0, CPP_DIR)

    import linecounter_cpp  # linecounter_cpp.pyd
    _HAS_CPP = True
except Exception:
    linecounter_cpp = None
    _HAS_CPP = False


class LineCounter:
    """
    main.py aynı kalsın diye dışarıya aynı isimle LineCounter veriyoruz.
    İçeride C++ varsa C++ çalışır.
    """

    def __init__(self, line_y, offset=8, direction="both", cooldown_frames=12):
        self.line_y = int(line_y)
        self.offset = int(offset)
        self.direction = direction
        self.cooldown_frames = int(cooldown_frames)

        if _HAS_CPP:
            self._impl = linecounter_cpp.LineCounterCpp(
                self.line_y, self.offset, self.direction, self.cooldown_frames
            )
        else:
            # Python fallback
            self.total = 0
            self.up = 0
            self.down = 0
            self.prev_y = {}
            self.cooldown = {}

    def update(self, id_centroids):
        """
        id_centroids: [(obj_id, cx, cy), ...]
        """
        if _HAS_CPP:
            # C++: tuple(int,float,float) list bekliyor
            total = self._impl.update([(int(i), float(cx), float(cy)) for (i, cx, cy) in id_centroids])
            # main.py'nin counter.up / counter.down erişimi için:
            self.total = self._impl.total
            self.up = self._impl.up
            self.down = self._impl.down
            return total

        # ---- Python fallback ----
        # cooldown azalt
        to_del = []
        for obj_id in list(self.cooldown.keys()):
            self.cooldown[obj_id] -= 1
            if self.cooldown[obj_id] <= 0:
                to_del.append(obj_id)
        for obj_id in to_del:
            del self.cooldown[obj_id]

        def in_band(y):
            return (self.line_y - self.offset) <= y <= (self.line_y + self.offset)

        for obj_id, cx, cy in id_centroids:
            obj_id = int(obj_id)
            cy = float(cy)

            prev = self.prev_y.get(obj_id, None)
            self.prev_y[obj_id] = cy

            if prev is None:
                continue
            if obj_id in self.cooldown:
                continue
            if not in_band(cy):
                continue

            moved_down = (cy > prev)
            moved_up = (cy < prev)

            if self.direction == "down" and not moved_down:
                continue
            if self.direction == "up" and not moved_up:
                continue

            self.total += 1
            if moved_down:
                self.down += 1
            elif moved_up:
                self.up += 1

            self.cooldown[obj_id] = self.cooldown_frames

        return self.total
