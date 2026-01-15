import cv2
import time

def main():
    print("Starting camera test...")
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Windows için daha stabil

    if not cap.isOpened():
        print("Kamera açılamadı")
        input("Enter'a bas ve çık...")
        return

    print("Kamera açıldı. Pencereyi görmüyorsan ALT+TAB ile ara.")

    cv2.namedWindow("Camera Test", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame okunamadı")
            break

        cv2.imshow("Camera Test", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("q basıldı, çıkılıyor...")
            break

        time.sleep(0.001)

    cap.release()
    cv2.destroyAllWindows()
    input("Bitti. Enter'a bas ve kapat...")

if __name__ == "__main__":
    main()





