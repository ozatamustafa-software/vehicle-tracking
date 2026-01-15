import cv2

class FrameProcessor:
    def __init__(self, width=640, height=480, blur_kernel=(5, 5), use_blur=True):
        self.width = width
        self.height = height
        self.blur_kernel = blur_kernel
        self.use_blur = use_blur

    def process(self, frame):
        frame = cv2.resize(frame, (self.width, self.height))
        if self.use_blur:
            frame = cv2.GaussianBlur(frame, self.blur_kernel, 0)
        return frame
