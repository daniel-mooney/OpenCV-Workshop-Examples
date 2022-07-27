import cv2 as cv
import matplotlib.pyplot as plt

def main() -> None:
    img = cv.imread("Images/flower.jpg")
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    laplacian = cv.Laplacian(img_gray, cv.CV_64F, ksize=5)
    sobel_x = cv.Sobel(img_gray, cv.CV_64F, 1,0, ksize=5)
    sobel_y = cv.Sobel(img_gray, cv.CV_64F, 0,1, ksize=5)

    plt.subplot(221)
    plt.imshow(img_gray, cmap="gray"), plt.title("Gray Scale")
    plt.xticks([]), plt.yticks([])
    
    plt.subplot(222)
    plt.imshow(laplacian, cmap="gray"), plt.title("Laplacian")
    plt.xticks([]), plt.yticks([])

    plt.subplot(223)
    plt.imshow(sobel_x, cmap="gray"), plt.title("Sobel X")
    plt.xticks([]), plt.yticks([])

    plt.subplot(224)
    plt.imshow(sobel_y, cmap="gray"), plt.title("Sobel Y")
    plt.xticks([]), plt.yticks([])

    plt.tight_layout()
    plt.suptitle("Gradient Algorithms")
    plt.show()

if __name__ == "__main__":
    main()