import cv2 as cv

def main() -> None:
    file_path = "Videos\people_walking.mp4"
    capture = cv.VideoCapture()

    while True:
        retval, frame = capture.read()

        if not retval:
            break

        cv.imshow("People Walking", frame)

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