import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def main() -> None:
    garden = cv.imread("..\Images\Outdoor-Toy-Storage.jpg")

    avg_blur = cv.blur(garden, (7, 7))
    gaussian = cv.GaussianBlur(garden, (7, 7), 0)       # ksize must have odd values
    median = cv.medianBlur(garden, 7)
    bilateral = cv.bilateralFilter(garden, 7, 75, 75)

    # Plot blurs
    plt.subplot(231), plt.imshow(garden), plt.title("Original")
    plt.xticks([]), plt.yticks([])

    plt.subplot(232), plt.imshow(avg_blur), plt.title("Average")
    plt.xticks([]), plt.yticks([])

    plt.subplot(233), plt.imshow(gaussian), plt.title("Gaussian")
    plt.xticks([]), plt.yticks([])

    plt.subplot(234), plt.imshow(median), plt.title("Median")
    plt.xticks([]), plt.yticks([])

    plt.subplot(235), plt.imshow(bilateral), plt.title("Bilateral")
    plt.xticks([]), plt.yticks([])

    plt.suptitle("Types of Blur")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()