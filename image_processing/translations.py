import cv2 as cv
import numpy as np

def rotate(img: cv.Mat, angle: float) -> cv.Mat:
    """
    Rotates an image about it's centre without scalling.
    Returns the rotated image.
    """
    rows, cols = img.shape[:2]
    img_centre = (rows-1) // 2, (cols-1) // 2

    rotM = cv.getRotationMatrix2D(img_centre, angle, 1)
    return cv.warpAffine(img, rotM, (cols,rows))

def translate(img: cv.Mat, shift: tuple[int, int]) -> cv.Mat:
    """
    Translates an image right by `x` and down by `y` for shift `(x, y)`.
    Returns the translated image
    """
    rows, cols = img.shape[:2]
    x, y = shift

    # Create translation matrix
    trans_matrix = np.float32([ [1, 0, x],
                                [0, 1, y]])
    return cv. warpAffine(img, trans_matrix, (cols, rows))      # warpAffine has cols, rows reversed

def main() -> None:
    beach = cv.imread("..\Images\Beach.jpg")

    rot_beach = rotate(beach, 90)
    trans_beach = translate(beach, (150, 20))

    cv.imshow("Beach", rot_beach)
    cv.imshow("translated", trans_beach)
    cv.waitKey(0)

if __name__ == "__main__":
    main()