import cv2 as cv
import matplotlib.pyplot as plt

def rescale(img: cv.Mat, scale: float) -> cv.Mat:
    height = int(img.shape[0] * scale)
    width = int(img.shape[1] * scale)

    return cv.resize(img, (width, height), interpolation=cv.INTER_AREA)

def main() -> None:
    img = cv.imread("..\Images\Beach.jpg")
    hsv_img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # Bounds for sand in HSV format
    sand_min = (12, 70, 250)
    sand_max = (18, 110, 255)

    # Create Threshold image
    sand_thresh = cv.inRange(hsv_img, sand_min, sand_max)

    # Apply two iterations of closing and one iteration of eroding morphological transformations
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (4, 4))                 # Ellipse with diameters (a, b)
    sand_closed = cv.morphologyEx(sand_thresh, cv.MORPH_CLOSE, kernel, iterations=2)
    sand_final_morph = cv.erode(sand_closed, kernel)

    # Rescale and show images
    scale_factor = 0.6
    sand_thresh = rescale(sand_thresh, scale_factor)
    sand_final_morph = rescale(sand_final_morph, scale_factor)
    img = rescale(img, scale_factor)

    cv.imshow("Sand Threshold", sand_thresh)
    cv.imshow("Morphological", sand_final_morph)
    cv.imshow("Original", img)

    cv.waitKey(0)

if __name__ == "__main__":
    main()