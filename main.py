import os
import sys

# C++ modül yolu (önce)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "cpp"))

import cv2
import time
import json

# C++ modül import (varsa)
try:
    import linecounter_cpp
    HAS_CPP_COUNTER = True
except Exception:
    linecounter_cpp = None
    HAS_CPP_COUNTER = False

from core.video_source import VideoSource
from core.frame_processor import FrameProcessor
from core.roi_selector import ROISelector

# debug amaçlı istersen açık tut (AMA TEK PENCERE İSTEĞİN İÇİN İÇERİDE KULLANMAYACAĞIZ)
from core.bg_subtractor import BGSubtractor
from core.morphology import MorphologyFilter

from core.yolo_detector import YOLODetector
from core.centroid_tracker import CentroidTracker

# Python LineCounter (C++ yoksa fallback)
from core.line_counter import LineCounter as PyLineCounter

from config.settings import AppConfig
from utils.fps import FPSMeter
from utils.overlay import draw_hud


# ----------------------------
# Metrics Logger (Task 15)
# ----------------------------
class MetricsLogger:
    """
    Basit metrics logger:
    - JSONL formatında log yazar (her satır bir json objesi)
    - Varsayılan: logs/metrics.jsonl
    """
    def __init__(self, out_path="logs/metrics.jsonl"):
        self.out_path = out_path
        os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def log(self, payload: dict):
        with open(self.out_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def draw_tracks(view, tracks, color=(0, 255, 0)):
    """
    tracks formatı:
    A) dict: {obj_id: (cx, cy)}
    B) dict: {obj_id: {"bbox":(x1,y1,x2,y2),"centroid":(cx,cy)}}
    C) list: [{"id":..,"bbox":..,"centroid":..}, ...]
    """
    if isinstance(tracks, dict):
        for obj_id, val in tracks.items():
            if isinstance(val, dict):
                bbox = val.get("bbox", None)
                c = val.get("centroid", None)
            else:
                bbox = None
                c = val

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)

            if c is not None:
                cx, cy = c
                cv2.circle(view, (int(cx), int(cy)), 4, color, -1)
                cv2.putText(
                    view, f"ID:{obj_id}",
                    (int(cx) + 6, int(cy) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )

    elif isinstance(tracks, list):
        for t in tracks:
            obj_id = t.get("id", "?")
            bbox = t.get("bbox", None)
            c = t.get("centroid", None)

            if bbox is not None:
                x1, y1, x2, y2 = bbox
                cv2.rectangle(view, (x1, y1), (x2, y2), color, 2)

            if c is not None:
                cx, cy = c
                cv2.circle(view, (int(cx), int(cy)), 4, color, -1)
                cv2.putText(
                    view, f"ID:{obj_id}",
                    (int(cx) + 6, int(cy) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
                )


def extract_boxes_from_yolo(dets):
    """
    YOLO çıktısı format farklarına dayanıklı bbox çıkarma.
    Dönen: [(x1,y1,x2,y2), ...]
    """
    boxes = []
    if dets is None:
        return boxes

    for d in dets:
        if isinstance(d, (tuple, list)) and len(d) == 4:
            x1, y1, x2, y2 = d
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
            continue

        if not isinstance(d, dict):
            continue

        if "xyxy" in d and d["xyxy"] is not None:
            x1, y1, x2, y2 = d["xyxy"]
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
            continue

        if "bbox" in d and d["bbox"] is not None:
            x1, y1, x2, y2 = d["bbox"]
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
            continue

        if "box" in d and d["box"] is not None:
            x1, y1, x2, y2 = d["box"]
            boxes.append((int(x1), int(y1), int(x2), int(y2)))
            continue

        if "xywh" in d and d["xywh"] is not None:
            x, y, w, h = d["xywh"]
            x1, y1, x2, y2 = int(x), int(y), int(x + w), int(y + h)
            boxes.append((x1, y1, x2, y2))
            continue

    return boxes


def main():
    cfg = AppConfig()

    video = VideoSource(source=cfg.source)
    processor = FrameProcessor(
        width=cfg.width,
        height=cfg.height,
        use_blur=cfg.use_blur,
        blur_kernel=cfg.blur_kernel
    )
    roi_selector = ROISelector(x=cfg.roi_x, y=cfg.roi_y, w=cfg.roi_w, h=cfg.roi_h)

    # (Debug modülleri var ama TEK PENCERE için imshow yapmayacağız)
    bg = BGSubtractor(
        history=cfg.mog2_history,
        var_threshold=cfg.mog2_var_threshold,
        detect_shadows=cfg.mog2_detect_shadows
    )
    morph = MorphologyFilter(kernel_size=cfg.morph_kernel_size, iterations=cfg.morph_iterations)

    # Sadece 4 teker motorlu: car=2, bus=5, truck=7
    yolo = YOLODetector(
        model_name=getattr(cfg, "yolo_model", "yolov8n.pt"),
        conf=getattr(cfg, "yolo_conf", 0.35),
        classes=[2, 5, 7],
        device=getattr(cfg, "yolo_device", None),
    )

    tracker = CentroidTracker(
        max_disappeared=getattr(cfg, "trk_max_disappeared", 20),
        max_distance=getattr(cfg, "trk_max_distance", 60),
    )

    line_y = getattr(cfg, "count_line_y", None)
    if line_y is None:
        line_y = int(getattr(cfg, "roi_h", cfg.height) * 0.5)

    # --- Counter seçimi: C++ varsa onu kullan ---
    if HAS_CPP_COUNTER and hasattr(linecounter_cpp, "LineCounter"):
        counter = linecounter_cpp.LineCounter(
            int(line_y),
            int(getattr(cfg, "count_line_offset", 8)),
            str(getattr(cfg, "count_direction", "both")),
            int(getattr(cfg, "count_cooldown_frames", 12)),
        )
        using_counter = "C++"
    else:
        counter = PyLineCounter(
            line_y=line_y,
            offset=getattr(cfg, "count_line_offset", 8),
            direction=getattr(cfg, "count_direction", "both"),
            cooldown_frames=getattr(cfg, "count_cooldown_frames", 12),
        )
        using_counter = "PY"

    fpsm = FPSMeter(avg_window=30)

    # ---- Task 15: metrics logger ----
    metrics_path = getattr(cfg, "metrics_path", "logs/metrics.jsonl")
    metrics_every = int(getattr(cfg, "metrics_every_n_frames", 30))
    metrics = MetricsLogger(out_path=metrics_path)

    # --- Tek pencere + daha sade HUD ayarları ---
    show_boxes = bool(getattr(cfg, "show_boxes", True))     # bbox+id çizilsin mi
    show_line = bool(getattr(cfg, "show_line", False))      # ✅ çizgi görünmesin (ama sayım çalışır)
    show_count_on_screen = bool(getattr(cfg, "show_count_on_screen", True))  # sayı ekranda yazsın mı

    # --- Pencereyi büyük açma ayarları ---
    window_name = "Vehicle Counter (Single Window)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 1) Büyük boyut (ekran çözünürlüğüne göre ayarla)
    # Not: Tam ekran istersen alttaki fullscreen satırını True yap.
    fullscreen = bool(getattr(cfg, "fullscreen", False))
    if fullscreen:
        cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    else:
        win_w = int(getattr(cfg, "window_w", 1280))
        win_h = int(getattr(cfg, "window_h", 720))
        cv2.resizeWindow(window_name, win_w, win_h)

    print(f"✅ Started (Hybrid + Task 15 | Single Window | Counter:{using_counter}). Press 'q' to quit.")
    frame_id = 0
    start_ts = time.time()

    while True:
        frame = video.read()
        if frame is None:
            print("❌ Frame alınamadı.")
            break

        if cfg.frame_skip > 0 and (frame_id % (cfg.frame_skip + 1) != 0):
            frame_id += 1
            continue
        frame_id += 1

        processed = processor.process(frame)

        roi_masked, roi_cropped, debug = roi_selector.apply(processed)
        work_frame = roi_cropped if cfg.use_roi_cropped else roi_masked

        # YOLO detect
        dets = yolo.detect(work_frame)
        boxes = extract_boxes_from_yolo(dets)

        # Track
        tracks = tracker.update(boxes)

        # Counter input: [(id,cx,cy), ...]
        id_centroids = []
        if isinstance(tracks, dict):
            for obj_id, val in tracks.items():
                if isinstance(val, dict):
                    c = val.get("centroid", None)
                else:
                    c = val
                if c is not None:
                    cx, cy = c
                    id_centroids.append((obj_id, cx, cy))
        elif isinstance(tracks, list):
            for t in tracks:
                obj_id = t.get("id", None)
                c = t.get("centroid", None)
                if obj_id is not None and c is not None:
                    cx, cy = c
                    id_centroids.append((obj_id, cx, cy))

        # --- Count (çizgi görünmese bile line_y ile sayım yapılır) ---
        total = counter.update(id_centroids)

        # --- View (TEK PENCERE) ---
        view = work_frame.copy()

        # Çizgi görünmesin ama istersen açabilirsin
        if show_line:
            cv2.line(view, (0, int(line_y)), (view.shape[1], int(line_y)), (0, 255, 255), 2)

        if show_boxes:
            draw_tracks(view, tracks)

        fpsm.tick()
        fps_val = fpsm.fps()

        if show_count_on_screen:
            # Up/Down yok → sadece total
            fps_text = f"FPS: {fps_val:.1f}"
            mode_text = "ROI: CROPPED" if cfg.use_roi_cropped else "ROI: MASKED"
            count_text = f"COUNT: {int(total)}"
            draw_hud(view, [fps_text, mode_text, count_text])

        cv2.imshow(window_name, view)

        # ---- Task 15: metrics logging ----
        if metrics_every > 0 and (frame_id % metrics_every == 0):
            up_val = getattr(counter, "up", None)
            down_val = getattr(counter, "down", None)

            payload = {
                "ts": time.time(),
                "elapsed_s": round(time.time() - start_ts, 3),
                "frame_id": frame_id,
                "fps": round(float(fps_val), 2),
                "roi_mode": "cropped" if cfg.use_roi_cropped else "masked",
                "yolo_conf": float(getattr(cfg, "yolo_conf", 0.35)),
                "count_total": int(total),
                "count_up": int(up_val) if up_val is not None else None,
                "count_down": int(down_val) if down_val is not None else None,
                "active_tracks": int(len(tracks)) if hasattr(tracks, "__len__") else None,
                "num_boxes": int(len(boxes)),
                "counter_impl": using_counter,
            }
            metrics.log(payload)

        if cv2.waitKey(cfg.wait_ms) & 0xFF == ord("q"):
            break

    video.release()
    cv2.destroyAllWindows()
    print("🛑 Closed.")


if __name__ == "__main__":
    main()
