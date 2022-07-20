import cv2 as cv
import matplotlib.pyplot as plt

def rescale(frame: cv.Mat, scale: float) -> cv.Mat:
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)
    dim = (width, height)

    return cv.resize(frame, dim, interpolation=cv.INTER_AREA)

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
    plt.imshow(orange_thresh, cmap="gray"), plt.title("Orange Threshold")
    plt.xticks([]), plt.yticks([])

    plt.suptitle("Colour Thresholding")
    plt.tight_layout()
    plt.show()

    return None

def basic_thresh() -> None:
    img = cv.imread("Images\Outdoor-Toy-Storage.jpg", cv.IMREAD_GRAYSCALE)

    retval, thresh = cv.threshold(img, 200, 255, cv.THRESH_BINARY)
    
    plt.subplot(121)
    plt.imshow(img, cmap="gray"), plt.title("Gray Scale")
    plt.xticks([]), plt.yticks([])

    plt.subplot(122)
    plt.imshow(thresh, cmap="gray"), plt.title("Threshold")
    plt.xticks([]), plt.yticks([])

    plt.tight_layout()
    plt.show()

    return None

def duck() -> None:
    duck = cv.imread("Images\giant_duck.jpg")
    duck_hsv = cv.cvtColor(duck, cv.COLOR_BGR2HSV)

    min_val = (15, 100, 100)
    max_val = (35, 255, 255)

    duck_thresh = cv.inRange(duck_hsv, min_val, max_val)
    cv.imshow("Duck Threshold", rescale(duck_thresh, 0.2))

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
    image()