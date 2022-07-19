import cv2 as cv

def main() -> None:
    capture = cv.VideoCapture(1)        # Front camera

    while True:
        retval, frame = capture.read()
        frame = cv.flip(frame, 1)        

        smoothed_frame = cv.GaussianBlur(frame, (5, 5), 0)
        
        cv.imshow("Video Capture", smoothed_frame)
        cv.imshow("Unblurred", frame)

        if cv.waitKey(1) == ord('d'):
            break
    
    capture.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    main()