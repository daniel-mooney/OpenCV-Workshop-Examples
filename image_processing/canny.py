import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np


def canny_threshold_values(img: cv.Mat, deviation: float=0.33) -> tuple[float, float]:
    """
    Recommended values for deviation: 0.34 <= d <= 0.5. By default, deviation is set to 0.33.
    Returns a tuple of (minVal, maxVal)
    """
    if len(img.shape) > 2:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    avgIntense = np.median(img)

    minVal = avgIntense * (1 - deviation)
    maxVal = avgIntense * (1 + deviation)

    return minVal, maxVal

def image() -> None:
    garden = cv.imread("..\Images\Outdoor-Toy-Storage.jpg")
    filtered_garden = cv.GaussianBlur(garden, (7, 7), 0)        # Filter noise using blur

    minVal, maxVal = canny_threshold_values(filtered_garden, 0.34)
    canny = cv.Canny(filtered_garden, minVal, maxVal)

    plt.subplot(121), plt.imshow(garden), plt.title("Original"), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(canny), plt.title("Canny"), plt.xticks([]), plt.yticks([])

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
    # image()
    video()