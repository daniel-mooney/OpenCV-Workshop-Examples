import cv2 as cv
import numpy as np

def shi_tomasi(img: cv.Mat, corners: int, quality: float, min_dist: int) -> cv.Mat:
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    corners = cv.goodFeaturesToTrack(img_gray, corners, quality, min_dist)      # run shi-tomasi

    # blank = np.zeros(img.shape)

    # Draw corners onto image
    marker_colour = [0, 255, 0]     # Green

    for corner in corners:
        x, y = np.int_(corner).ravel()       # x, y co-ordinate of corner
        cv.circle(img, (x, y), 3, marker_colour, -1)
    
    return img

def video() -> None:
    capture = cv.VideoCapture(1)        # front cam

    while True:
        _, frame = capture.read()
        frame = cv.flip(frame, 1)       # flip horizontally

        corners = shi_tomasi(frame, 1000, 0.01, 10)
        cv.imshow("Corner Detection", corners)

        if cv.waitKey(1) == ord('d'):
            break
    
    capture.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    video()