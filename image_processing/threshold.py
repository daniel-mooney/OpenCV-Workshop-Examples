import cv2 as cv
import matplotlib.pyplot as plt

def image() -> None:
    img = cv.imread("Images\Outdoor-Toy-Storage.jpg")
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)                        # Convert image to HSV image
    rgb_img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

    # Bounds for the colour orange in HSV format
    orange_min = (10, 100, 200)
    orange_max = (20, 255, 255)

    orange_thresh = cv.inRange(hsv_img, orange_min, orange_max)         # Create Threshold
    
    plt.subplot(121)
    plt.imshow(rgb_img), plt.title("Original")
    plt.xticks([]), plt.yticks([])

    plt.subplot(122)
    plt.imshow(orange_thresh), plt.title("Orange Threshold")
    plt.xticks([]), plt.yticks([])

    plt.suptitle("Colour Thresholding")
    plt.tight_layout()
    plt.show()

    return None

def basic_thresh() -> None:
    img = cv.imread("Images\Outdoor-Toy-Storage.jpg", cv.IMREAD_GRAYSCALE)

    retval, thresh = cv.threshold(img, 200, 255, cv.THRESH_BINARY)
    cv.imshow("Threshold", thresh)

    cv.waitKey(0)
    return None

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

    return None

if __name__ == "__main__":
    basic_thresh()