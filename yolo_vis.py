import cv2


def draw_detections(frame, dets, show_label=True):
    """
    frame: BGR image (numpy)
    dets: [{"xyxy":(...), "conf":float, "name":str}, ...]
    """
    out = frame.copy()
    for d in dets:
        x1, y1, x2, y2 = d["xyxy"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)

        if show_label:
            label = f'{d["name"]} {d["conf"]:.2f}'
            cv2.putText(out, label, (x1, max(0, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return out


def det_centroid(det):
    x1, y1, x2, y2 = det["xyxy"]
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    return (cx, cy)
