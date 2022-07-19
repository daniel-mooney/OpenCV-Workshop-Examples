import cv2 as cv

def main():
    videoPath = "../Videos/Tom_Scott_Choking.mp4"      # Set 0 from back cam, 1 for front cam
    capture = cv.VideoCapture(videoPath)

    while True:
        retval, img = capture.read()        # retval is bool for successful read

        # Reset video once end is reached
        if not retval:
            capture = cv.VideoCapture(videoPath)
            continue
        
        cv.imshow("Tom Scott", img)
        
        if cv.waitKey(17) == ord('d'):
            break
    
    capture.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    main()