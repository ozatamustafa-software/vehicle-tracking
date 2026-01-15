import cv2

def draw_hud(img, text_lines, org=(10, 25), line_height=25):
    x, y = org
    for i, t in enumerate(text_lines):
        cv2.putText(
            img,
            t,
            (x, y + i * line_height),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return img
