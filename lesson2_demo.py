import cv2 as cv
import numpy as np

def main() -> None:
    image = cv.imread("Images\Outdoor-Toy-Storage.jpg")

    img_blurred = cv.blur(image, )

    cv.imshow("Backyard", image)
    cv.waitKey(0)

if __name__ == "__main__":
    main()