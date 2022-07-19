import cv2 as cv

def image() -> None:
    img = cv.imread("../Images/Outdoor-Toy-Storage.jpg")
    grayscale = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    cv.imshow("Gray", grayscale)
    cv.imshow("Colour", img)
    cv.waitKey(0)

def video() -> None:
    capture:video = cv.VideoCapture(1)        # Front cam

    while True:
        _, frame = capture.read()                       # Read frame
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)    # Convert to grayscale
        flipped_gray = cv.flip(gray, 1)                 # Flip image

        cv.imshow("Gray me", flipped_gray)
        cv.imshow("Unflipped", gray)

        if cv.waitKey(1) == ord('d'):
            break
    
    capture.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    image()
    video()