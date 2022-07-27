import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def canny_threshold_values(img: cv.Mat, deviation: float=0.33) -> tuple[float, float]:   
    avgIntense = np.median(img)

    minVal = avgIntense * (1 - deviation)
    maxVal = avgIntense * (1 + deviation)

    return minVal, maxVal

def image() -> None:
    img = cv.imread("../Images/dog.jpg")
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img_gray = cv.GaussianBlur(img_gray, (5,5), 0)

    minVal, maxVal = canny_threshold_values(img_gray)
    canny = cv.Canny(img_gray, minVal, maxVal)

    img_rgb = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    plt.subplot(121), plt.imshow(img_rgb), plt.title("Original"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(canny, cmap="gray"), plt.title("Canny Edge Detector"), plt.xticks([]), plt.yticks([])

    plt.suptitle("Edge Detection")
    plt.tight_layout()
    plt.show()

def video() -> None:
    capture = cv.VideoCapture(1)        # Front camera

    while True:
        _, frame = capture.read()
        frame = cv.flip(frame, 1)

        filtered_frame = cv.GaussianBlur(frame, (7, 7), 0)
        minVal, maxVal = canny_threshold_values(filtered_frame)

        canny = cv.Canny(filtered_frame, minVal, maxVal)
        cv.imshow("CannyVideo", canny)

        if cv.waitKey(1) == ord('d'):
            break
    
    capture.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    image()
    # video()