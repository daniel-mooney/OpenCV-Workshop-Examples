from canny import canny_threshold_values
import cv2 as cv

def rescale(img: cv.Mat, scale: float) -> cv.Mat:
    height = int(img.shape[0] * scale)
    width = int(img.shape[1] * scale)
    dim = (width, height)

    return cv.resize(img, dim, interpolation=cv.INTER_AREA)    

def image() -> None:
    flower = cv.imread("../Images/flower.jpg", 0)
    scaled_flower = rescale(flower, 0.35)            # half image size
    
    blurred_flower = cv.GaussianBlur(scaled_flower, (7, 7), 0)
    sobel_x = cv.Sobel(blurred_flower, cv.CV_64F, 1, 0, ksize=5)
    sobel_y = cv.Sobel(blurred_flower, cv.CV_64F, 0, 1, ksize=5)
    combined_sobel = cv.bitwise_or(sobel_x, sobel_y)
    
    # Canny
    minVal, maxVal = canny_threshold_values(blurred_flower)
    canny = cv.Canny(blurred_flower, minVal, maxVal)    

    cv.imshow("Original", scaled_flower)
    cv.imshow("Sobel X", sobel_x)
    cv.imshow("Sobel Y", sobel_y)
    cv.imshow("Combined Sobel", combined_sobel)
    cv.imshow("Canny", canny)

    cv.waitKey(0)

if __name__ == "__main__":
    image()