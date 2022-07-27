from canny import canny_threshold_values
import cv2 as cv
import numpy as np

def rescale(frame, scale):
    height = int(frame.shape[0] * scale)
    width = int(frame.shape[1] * scale)

    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)

def video() -> None:
    capture = cv.VideoCapture(1)        # front cam

    while True:
        _, img = capture.read()
        img = cv.flip(img, 1)

        img_blurred = cv.GaussianBlur(img, (7, 7), 0)
        img_gray = cv.cvtColor(img_blurred, cv.COLOR_BGR2GRAY)

        min_val, max_val = canny_threshold_values(img_blurred, 0.4)
        img_canny = cv.Canny(img_gray, min_val, max_val)
        contours, hierachy = cv.findContours(img_canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

        # Draw images
        blank = np.zeros(img_canny.shape)
        cv.drawContours(blank, contours, len(contours) - 1, (0, 0, 255), 1)

        cv.imshow("Original", img_blurred)
        cv.imshow("Contours", blank)
        cv.imshow("Canny", img_canny)

        if cv.waitKey(1) == ord('d'):
            break
    
    capture.release()
    cv.destroyAllWindows()

def filter_contours(min_area: int, contours: list) -> list:
    filtered_contours = []

    for contour in contours:
        if cv.contourArea(contour) > min_area:
            filtered_contours.append(contour)
    
    return filtered_contours

def draw_bounding_rectangles(image, contours) -> cv.Mat:
    colour = (0,150, 100)
    image_copy = image.copy()

    for contour in contours:
        x,y,w,h = cv.boundingRect(contour)
        cv.rectangle(image_copy, (x,y), (x+w, y+h), colour, 2)
    
    return image_copy    

def image() -> None:
    img = cv.imread("Images/ball.jpg")
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    img_blurred = cv.GaussianBlur(img_gray, (5,5), 0)

    min_val, max_val = canny_threshold_values(img_blurred)
    img_canny = cv.Canny(img_blurred, min_val, max_val)

    contours, _ = cv.findContours(img_canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    blank = np.zeros(img.shape[:2])
    img_contours = cv.drawContours(blank, contours, -1, (255,0,0), 2)
    img_boundings = draw_bounding_rectangles(img, contours)

    cv.imshow("Contours", rescale(img_contours, 0.6))
    cv.imshow("Bounding", rescale(img_boundings, 0.6))
    print(len(contours))

    cv.waitKey(0)

if __name__ == "__main__":
    # video()
    image()