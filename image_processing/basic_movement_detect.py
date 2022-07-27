import cv2 as cv

def rescale(image: cv.Mat, scale: float) -> cv.Mat:
    height = int(image.shape[0] * scale)
    width = int(image.shape[1] * scale)

    return cv.resize(image, (width, height), interpolation=cv.INTER_AREA)

def filter_contours(contours, min_area) -> list:
    filtered_contours = []

    for cnt in contours:
        if cv.contourArea(cnt) > min_area:
            filtered_contours.append(cnt)

    return filtered_contours

def get_contours(image, min_area) -> list:
    contours, hierachy = cv.findContours(image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    contours = filter_contours(contours, min_area)

    return contours

def draw_bounding_rect(image, contours) -> cv.Mat:
    image_copy = image.copy()
    colour = (0,255,0)

    for cnt in contours:
        x, y, w, h = cv.boundingRect(cnt)
        cv.rectangle(image_copy, (x,y), (x+w, y+h), colour, 2)
    
    return image_copy

def main() -> None:
    file_path = "Videos\people_walking.mp4"
    capture = cv.VideoCapture(file_path)

    background_sub = cv.createBackgroundSubtractorMOG2()

    while True:
        retval, frame = capture.read()

        if not retval:
            break

        bg_mask = background_sub.apply(frame)
        retval, bg_mask = cv.threshold(bg_mask, 200, 255, cv.THRESH_BINARY)

        contours = get_contours(bg_mask, 1000)

        cv.imshow("Background Mask", rescale(bg_mask, 0.7))
        cv.imshow("People Walking", rescale(frame, 0.7))

        key = cv.waitKey(17)
        if key == ord('p'):
            key = cv.waitKey(0)
        if key == ord('d'):
            break
    
    capture.release()
    cv.destroyAllWindows()

    return None

if __name__ == "__main__":
    main()