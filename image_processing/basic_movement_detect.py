import cv2 as cv

def rescale(image: cv.Mat, scale: float) -> cv.Mat:
    height = int(image.shape[:2] * scale)
    width = int(image.shape[:2] * scale)

    return cv.resize(image, (width, height), interpolation=cv.INTER_AREA)

def main() -> None:
    file_path = "Videos\people_walking.mp4"
    capture = cv.VideoCapture(file_path)

    background_sub = cv.createBackgroundSubtractorKNN(detectShadows=False)

    while True:
        retval, frame = capture.read()

        if not retval:
            break

        bg_mask = background_sub.apply(frame)

        cv.imshow("People Walking", bg_mask)

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