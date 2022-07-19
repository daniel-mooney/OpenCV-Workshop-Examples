from canny import canny_threshold_values
import cv2 as cv
import numpy as np

def video() -> None:
    capture = cv.VideoCapture(1)        # front cam

    while True:
        _, img = capture.read()
        img = cv.flip(img, 1)

        img_blurred = cv.GaussianBlur(img, (7, 7), 0)
        img_gray = cv.cvtColor(img_blurred, cv.COLOR_BGR2GRAY)

        min_val, max_val = canny_threshold_values(img_blurred, 0.4)
        img_canny = cv.Canny(img_gray, min_val, max_val)
        contours, hierachy = cv.findContours(img_canny, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        # Draw images
        blank = np.zeros(img_canny.shape)
        cv.drawContours(blank, contours, len(contours) - 1, (0, 0, 255), 1)

        cv.imshow("Original", img_blurred)
        cv.imshow("Contours", blank)
        cv.imshow("Canny", img_canny)

        if cv.waitKey(1) == ord('d'):
            break
    
    capture.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    video()