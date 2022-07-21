import cv2 as cv

def detect_objects(frame: cv.Mat, cascade) -> cv.Mat:
    """
    Returns a frame with bounding boxes around any detected faces.
    Detects any objects specified by the type of cascade passed as an argument.
    """
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)            # Increase image contrast
    frame_copy = frame.copy()

    # Detect objects
    objects = cascade.detectMultiScale(frame_gray)

    for (x,y,h,w) in objects:
        # x,y are the coordinates of the top left corner
        frame_copy = cv.rectangle(frame_copy, (x,y), (x+w, y+h), (0,255,0), 2)
    
    return frame_copy

def main() -> None:
    camera = cv.VideoCapture(1)     # front camera

    # Initialise cascade classifier object and load classifier file
    face_cascade = cv.CascadeClassifier()
    success = face_cascade.load("face_detection\Classifier_files\haarcascade_frontalface_alt.xml")

    if not success:
        print("Failed to open face cascade file")
        return None

    while True:
        retval, frame = camera.read()
        frame = cv.flip(frame, 1)

        if not retval:
            break

        face_frame = detect_objects(frame, face_cascade)
        
        cv.imshow("Face Detector", face_frame)

        if cv.waitKey(1) == ord('d'):
            break
    
    camera.release()
    cv.destroyAllWindows()
    return None

if __name__ == "__main__":
    main()