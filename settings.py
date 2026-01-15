from dataclasses import dataclass

@dataclass
class AppConfig:
    # Video
    source: int | str = 0
    width: int = 640
    height: int = 480

    # ROI (cropped önerilir)
    roi_x: int = 0
    roi_y: int = 200
    roi_w: int = 640
    roi_h: int = 280

    # Preprocess
    use_blur: bool = True
    blur_kernel: tuple[int, int] = (5, 5)

    # MOG2
    mog2_history: int = 300
    mog2_var_threshold: int = 50
    mog2_detect_shadows: bool = True

    # Morphology
    morph_kernel_size: int = 5
    morph_iterations: int = 1

    # Optimization (FPS / CPU)
    use_roi_cropped: bool = True     # True: sadece ROI alanında çalış
    show_debug_windows: bool = True  # False yaparsan tek pencere/az yük
    frame_skip: int = 0              # 0: her frame, 1: 1 frame atla, 2: 2 frame atla...
    wait_ms: int = 1                 # imshow için waitKey süresi
