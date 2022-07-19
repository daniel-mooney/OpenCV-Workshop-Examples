import cv2 as cv

def image() -> None:
    img = cv.imread("Images\Outdoor-Toy-Storage.jpg")
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)                        # Convert image to HSV image

    # Bounds for the colour orange in HSV format
    orange_min = (10, 100, 200)
    orange_max = (20, 255, 255)

    orange_thresh = cv.inRange(hsv_img, orange_min, orange_max)         # Create Threshold
    cv.imshow("Orange Threshold", orange_thresh)
    cv.imshow("Original", img)

    cv.waitKey(0)

def video() -> None:
    capture = cv.VideoCapture(1)        # front cam

    while True:
        _, frame = capture.read()
        frame = cv.cvtColor(cv.flip(frame, 1), cv.COLOR_BGR2GRAY)

        _, thresh = cv.threshold(frame, 80, 255, cv.THRESH_BINARY)
        cv.imshow("Threshold", thresh)

        if cv.waitKey(1) == ord('d'):
            break
    
    capture.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    image()