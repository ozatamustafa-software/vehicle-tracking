import cv2

class BGSubtractor:
    def __init__(self, history=300, var_threshold=50, detect_shadows=True):
        """
        MOG2 background subtractor
        - history: arka planı öğrenme süresi (frame sayısı)
        - var_threshold: hassasiyet (düşük -> daha hassas)
        - detect_shadows: gölgeleri ayırmaya çalışır (True olursa maskede gri tonlar görebilirsin)
        """
        self.sub = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )

    def apply(self, frame, learning_rate=-1):
        """
        frame: ROI görüntüsü
        learning_rate: -1 (otomatik) / 0 (öğrenme kapalı) / 0.001 gibi küçük değerler
        """
        fg_mask = self.sub.apply(frame, learningRate=learning_rate)
        return fg_mask
